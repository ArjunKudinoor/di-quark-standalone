#!/usr/bin/env python3
"""Fig. 3 pion extension; explicit copies of only the selection-dependent helpers.

All kinematics, color checks, PYTHIA cards and observables are imported unchanged
from jet_hadron_observables_split_pi_pm. Selection order is identical. Neutral
pions are read as physical parent four-vectors before their decay, without
changing decays or shower settings. Original modules are never patched.
"""
from __future__ import annotations
import math
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import jet_hadron_observables_split_pi_pm as original
from jet_hadron_observables_split_pi_pm import (
    pythia8, JH, FLIP_Z_PTREL, ZLC_MAX, ZLC_MIN, build_LT, flip_z, is_hadron,
    p3, extract_beams, row_p4, dis_xy_q_within_selection, final_colored_partons,
    color_balance_ok, flipped_lab_beam_arrays_from_event_row,
    kinematics_lab_to_breit_lt_from_flip_lab_beams,
    collider_forward_pion_pseudorapidity, pion_lab_eta_native_beam_from_breit,
    compute_observables_for_pion, build_pythia_reinjector,
)


def run_pythia_reinject_collect(p, partons, LT, csv_momenta_breit, *, include_pi0=True):
    ok, error, hadrons = original.run_pythia_reinject_collect(p, partons, LT, csv_momenta_breit)
    if not ok or not include_pi0:
        return ok, error, hadrons
    # Keep original event-record order, including pi0 before its decay. Exclude
    # intermediate record copies with a pi0 daughter to count each pi0 once.
    hadrons = []
    for i in range(p.event.size()):
        particle = p.event[i]
        pid = int(particle.id())
        if not particle.isHadron():
            continue
        if pid == 111:
            if any(p.event[j].id() == 111 for j in particle.daughterList()):
                continue
        elif not particle.isFinal():
            continue
        hadrons.append((pid, np.array([particle.e(), particle.px(), particle.py(), particle.pz()], dtype=float)))
    return ok, error, hadrons


def discard_pions_outside_collider_eta(
    hadrons: List[Tuple[int, np.ndarray]],
    LT_lab_to_breit: np.ndarray,
    eta_min: float,
    eta_max: float,
) -> List[Tuple[int, np.ndarray]]:
    """
    Drop charged/neutral pions unless ``eta_min < η < η_max`` with **proton-forward** η
    (:func:`collider_forward_pion_pseudorapidity`).

    Lab vector: ``P_beam_lab = flip_z(LT^{-1} P_h^{Breit}, FLIP_Z_PTREL)``.
    Returned entries are still Breit four-vectors for downstream observables.
    """
    LT = np.asarray(LT_lab_to_breit, dtype=np.float64)
    try:
        LT_inv = np.linalg.inv(LT)
    except np.linalg.LinAlgError:
        LT_inv = np.linalg.pinv(LT)
    out: List[Tuple[int, np.ndarray]] = []
    for pid, p4 in hadrons:
        ap = abs(int(pid))
        if ap in (211, 111) and is_hadron(int(pid)):
            p_b = np.asarray(p4, dtype=np.float64)
            p_flipped_lab = LT_inv @ p_b
            p_beam_lab = flip_z(p_flipped_lab, FLIP_Z_PTREL)
            try:
                eta = collider_forward_pion_pseudorapidity(p_beam_lab)
            except ValueError:
                continue
            if not (float(eta_min) < eta < float(eta_max)) or not math.isfinite(eta):
                continue
        out.append((int(pid), np.asarray(p4, dtype=np.float64).copy()))
    return out


def best_leading_pion_breit(
    hadrons: List[Tuple[int, np.ndarray]],
    *,
    breit_hemisphere: str = "target",
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Single leading charged or neutral π in a Breit hemisphere (post-injection hadron list):

    ``target``: max E among |pdg| in (211,111) with ``pz_Breit > 0`` (proton / target-fragmentation side).
    ``current``: max E among |pdg| in (211,111) with ``pz_Breit < 0`` (current hemisphere).

    Default ``target`` matches ``split_kinematics_extract.leading_target_pion_breit``.
    """
    if breit_hemisphere not in ("target", "current"):
        raise ValueError(
            f"breit_hemisphere must be 'target' or 'current', got {breit_hemisphere!r}"
        )
    best_e = -1.0
    best_pid: Optional[int] = None
    best_p4: Optional[np.ndarray] = None
    for pid, p4 in hadrons:
        ap = abs(int(pid))
        if ap not in (211, 111) or not is_hadron(int(pid)):
            continue
        p4 = np.asarray(p4, dtype=np.float64)
        E, pz = float(p4[0]), float(p4[3])
        if breit_hemisphere == "target":
            if pz <= 0:
                continue
        else:
            if pz >= 0:
                continue
        if E > best_e:
            best_e = E
            best_pid = int(pid)
            best_p4 = p4.copy()
    return best_pid, best_p4


def process_event_dataframe(
    df: pd.DataFrame,
    struck_idx: int,
    sample: str,
    csv_label: str,
    csv_momenta_breit: bool = False,
    pythia_reuse: Optional[pythia8.Pythia] = None,
    *,
    event_row: Optional[pd.Series] = None,
    include_pi0: bool = True,
    pion_collider_eta_acceptance: Optional[Tuple[float, float]] = None,
    pion_breit_hemisphere: str = "target",
    pythia_e_beam_gev: float = 18.0,
    pythia_p_beam_gev: float = 275.0,
    dis_x_exclusive_floor: Optional[float] = None,
    dis_q_open_interval_gev: Optional[Tuple[float, float]] = None,
    pion_pt_open_interval_gev: Optional[Tuple[float, float]] = None,
    jet_pt_open_interval_gev: Optional[Tuple[float, float]] = None,
    pythia_parton_level_isr: bool = True,
    pythia_parton_level_fsr: bool = True,
) -> List[Dict[str, Any]]:
    """Return exactly one row per event (leading eligible pion in chosen Breit hemisphere or failure).

    ``pion_breit_hemisphere``: ``"target"`` (``pz_Breit > 0``) or ``"current"`` (``pz_Breit < 0``).

    ``pythia_*_beam_gev``: passed to PYTHIA when building a fresh reinjector inside
    :func:`run_pythia_collect_final_hadrons`; if you pass ``pythia_reuse``, that instance must have
    been created with matching beam energies.

    ``dis_x_exclusive_floor``: if set, reject unless ``x > floor`` (still capped by ``xmax_ptrel``).
    ``dis_q_open_interval_gev``: if set ``(q_lo, q_hi)``, reject unless ``q_lo < Q < q_hi``.
    Defaults use ``xmin_ptrel`` / ``Qmin_ptrel`` / ``Qmax_ptrel``.
    ``pion_pt_open_interval_gev``: if set ``(lo, hi)``, require strict ``lo < P_hT < hi`` where
    ``P_hT = |(pion_px, pion_py)|`` in Breit.
    ``jet_pt_open_interval_gev``: if set ``(lo, hi)``, require strict ``lo < P_JT < hi`` where
    ``P_JT = |(k_out_px, k_out_py)|`` in Breit.
    ``pythia_parton_level_isr`` / ``pythia_parton_level_fsr``: only used when ``pythia_reuse``
    is None (fresh PYTHIA in :func:`run_pythia_collect_final_hadrons`). If ``pythia_reuse``
    is set, it must match these settings (caller responsibility).
    """
    df = df.sort_values("particle_index").reset_index(drop=True)
    base = {
        "event_id": int(df["event_id"].iloc[0]),
        "sample": sample,
        "csv": csv_label,
    }

    def _fail(reason: str, **extra: Any) -> List[Dict[str, Any]]:
        return [{**base, "pion": "", "pdg_id": 0, "ok": False, "reason": reason, **extra}]

    e_in, e_sc, p_in, beam_msg = extract_beams(df)
    if e_in is None:
        return _fail(beam_msg)

    k_row = df[df["particle_index"] == struck_idx]
    if len(k_row) != 1:
        return _fail("k_out_row")
    k_out = row_p4(k_row.iloc[0])

    if csv_momenta_breit:
        e_in_ev = e_in
        e_sc_ev = e_sc
        p_in_ev = p_in
        k_out_ev = k_out
    else:
        e_in_ev = flip_z(e_in, FLIP_Z_PTREL)
        e_sc_ev = flip_z(e_sc, FLIP_Z_PTREL)
        p_in_ev = flip_z(p_in, FLIP_Z_PTREL)
        k_out_ev = flip_z(k_out, FLIP_Z_PTREL)

    qmu = e_in_ev - e_sc_ev
    Q2 = -(qmu[0] * qmu[0] - qmu[1] * qmu[1] - qmu[2] * qmu[2] - qmu[3] * qmu[3])
    if Q2 <= 0:
        return _fail("Q2")
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(qmu[1], qmu[2]))
    p_dot_q = float(
        p_in_ev[0] * qmu[0] - p_in_ev[1] * qmu[1] - p_in_ev[2] * qmu[2] - p_in_ev[3] * qmu[3]
    )
    if p_dot_q == 0:
        return _fail("pdotq")
    x = Q2 / (2.0 * p_dot_q)
    if not dis_xy_q_within_selection(
        x,
        Q,
        x_exclusive_floor=dis_x_exclusive_floor,
        q_open_interval_gev=dis_q_open_interval_gev,
    ):
        return _fail("xQ_window")

    Ee = float(e_in_ev[0])
    Ep = float(p_in_ev[0])
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    phiq = float(np.arctan2(qmu[2], qmu[1]))
    if csv_momenta_breit:
        LT = np.eye(4, dtype=np.float64)
    else:
        LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
        if LT is None:
            return _fail("build_LT")

    partons = final_colored_partons(df)
    c_ok, c_msg = color_balance_ok(partons)
    if not c_ok:
        return _fail("color", color_message=c_msg)

    if pythia_reuse is None:
        pythia_reuse = build_pythia_reinjector(
            e_beam_gev=pythia_e_beam_gev, p_beam_gev=pythia_p_beam_gev,
            parton_level_isr=pythia_parton_level_isr, parton_level_fsr=pythia_parton_level_fsr,
        )
    py_ok, py_err, hadrons = run_pythia_reinject_collect(
        pythia_reuse, partons, LT, csv_momenta_breit, include_pi0=include_pi0
    )
    if not py_ok:
        return _fail("pythia", pythia_error=py_err)

    if not include_pi0:
        hadrons = [(pid, v) for pid, v in hadrons if pid != 111]

    if pion_collider_eta_acceptance is not None:
        emin, emax = float(pion_collider_eta_acceptance[0]), float(
            pion_collider_eta_acceptance[1]
        )
        if csv_momenta_breit:
            if event_row is None:
                return _fail("event_row_missing_for_eta")
            blt = flipped_lab_beam_arrays_from_event_row(event_row)
            if blt is None:
                return _fail("missing_lab_beams_for_eta")
            LT_eta, eta_err = kinematics_lab_to_breit_lt_from_flip_lab_beams(
                *blt,
                x_exclusive_floor=dis_x_exclusive_floor,
                q_open_interval_gev=dis_q_open_interval_gev,
            )
            if LT_eta is None:
                return _fail(eta_err or "build_LT_eta")
        else:
            LT_eta = LT
        hadrons = discard_pions_outside_collider_eta(
            hadrons, LT_eta, emin, emax
        )

    boost = lambda v: LT @ np.asarray(v, dtype=float)  # identity when csv_momenta_breit
    p_breit = boost(p_in_ev)
    if float(p_breit[0] + p_breit[3]) <= 0:
        return _fail("P_plus")

    # Jet pT: outgoing quark (same convention as analyze_jet_hadron_transverse_observables.py)
    k_out_breit = boost(k_out_ev)
    pT_jet = JH._pT_vec_breit(k_out_breit)
    pjt_mag = float(np.linalg.norm(pT_jet))

    k_in_old = k_out_ev - qmu
    p_rem = p_in_ev - k_in_old
    p_rem_breit = boost(p_rem)
    if float(np.linalg.norm(p3(p_rem_breit))) <= 0:
        return _fail("remnant_pt")
    q_breit = boost(qmu)

    win_pid, pib = best_leading_pion_breit(
        hadrons, breit_hemisphere=pion_breit_hemisphere
    )
    pion_tag = "piplus" if win_pid == 211 else "piminus" if win_pid == -211 else "pi0" if win_pid == 111 else ""
    r: Dict[str, Any] = {
        **base,
        "pion": pion_tag,
        "pdg_id": int(win_pid) if win_pid is not None else 0,
        "Q2": float(Q2),
        "xB": float(x),
        "color_ok": True,
        "pythia_ok": True,
        "n_final_hadrons": len(hadrons),
        "pion_lab_eta": float("nan"),
    }
    if pib is None:
        r.update({"ok": False, "reason": "no_pion_candidate"})
        return [r]
    r["pion_breit_energy_GeV"] = float(pib[0])
    r["pion_breit_pz_GeV"] = float(pib[3])
    eta_native = pion_lab_eta_native_beam_from_breit(
        pib,
        csv_momenta_breit=csv_momenta_breit,
        event_row=event_row,
        LT_breit_lab_to_process=LT,
        x_exclusive_floor=dis_x_exclusive_floor,
        q_open_interval_gev=dis_q_open_interval_gev,
    )
    if eta_native is not None:
        r["pion_lab_eta"] = float(eta_native)
    pht_mag = float(np.linalg.norm(JH._pT_vec_breit(pib)))
    r["pion_pht_GeV"] = pht_mag
    r["jet_pjt_GeV"] = pjt_mag
    if pion_pt_open_interval_gev is not None:
        p_lo, p_hi = float(pion_pt_open_interval_gev[0]), float(pion_pt_open_interval_gev[1])
        if not (p_lo < pht_mag < p_hi):
            r.update({"ok": False, "reason": "pht_window"})
            return [r]
    if jet_pt_open_interval_gev is not None:
        j_lo, j_hi = float(jet_pt_open_interval_gev[0]), float(jet_pt_open_interval_gev[1])
        if not (j_lo < pjt_mag < j_hi):
            r.update({"ok": False, "reason": "pjt_window"})
            return [r]
    phi_hj, dmag, pbar, sum_mag, zLC, msg = compute_observables_for_pion(
        pib, pT_jet, p_breit, q_breit
    )
    if msg != "ok":
        r.update({"ok": False, "reason": msg, "zLC": zLC})
        return [r]
    r.update(
        {
            "ok": True,
            "reason": "ok",
            "phi_hJ": float(phi_hj) if phi_hj is not None else float("nan"),
            "delta_P_hJ_GeV": dmag,
            "Pbar_t_GeV": pbar,
            "sum_mag_GeV": sum_mag,
            "zLC": zLC,
        }
    )
    return [r]

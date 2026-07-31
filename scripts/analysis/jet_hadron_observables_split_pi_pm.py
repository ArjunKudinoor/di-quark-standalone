#!/usr/bin/env python3
"""
Jet–hadron transverse observables for split DIS CSVs (unchanged vs altered) after
reinjection + hadronization, **one recorded hadron per event**: the leading charged
pion (same rule as ``split_kinematics_extract.leading_target_pion_breit``).

Frame: Breit (same LT construction as analyze_jet_hadron_transverse_observables.py).
Jet: outgoing struck quark k_out from the hard-event CSV row (particle_index = struck
outgoing), same frame as stored momenta; pT_jet = (P_x, P_y) of boost(k_out) in Breit.

Hadron: among final PYTHIA hadrons in the post-injection Breit frame with **|pdg| == 211**,
highest **E** in the target hemisphere (**pz > 0**). The output row uses ``pion`` =
``"piplus"`` or ``"piminus"`` according to the winner’s charge (for π⁺ / π⁻ comparison plots).

When that candidate exists and passes the z_LC cut:
  - phi_hJ: angle in [0, π] between pT_jet and pT_hadron in the Breit frame.
  - |Delta P_hJ| = |pT_jet - pT_hadron|  (GeV).
  - |Pbar_t| = |(pT_jet + pT_hadron) / 2| = (1/2) |pT_jet + pT_hadron|  (GeV).
  - sum_mag_GeV = |pT_jet + pT_hadron| — same as observable 2 in
    analyze_jet_hadron_transverse_observables.py (used for the sum-magnitude PDF).

Reuses (via import of analyze_jet_hadron_transverse_observables): ANGLE_BINS/RANGE,
SUM_DIFF_BINS/RANGE, _pT_vec_breit, _angle_between_2d, OUTPUT_PREFIX / HADRON_TAG /
FRAME_TAG for plot style. Does not call run_observables_for_label (that path reads
numpy shards, not reinjected CSVs).

z_LC is the paper's light-cone momentum fraction:
  z_LC = q · P_h / q · P_proton
computed in the Breit frame. Events are kept when z_LC > 0.2.

Workflow (``--csv-momenta-frame lab``, default): (1) CSV momenta are lab frame.
(2) Build LT from beams + q. (3) flip_z(lab) then LT @ v for each colored parton;
inject into PYTHIA. (4) Jet/remnant/q use the same LT on lab CSV vectors.

Workflow (``--csv-momenta-frame breit``): CSV is already DIS Breit (e.g.
``generate_dis_isr_parton_dataset.py`` default output). Do **not** apply flip_z or
build_LT to CSV rows; use four-vectors as-is and LT = I for boosts. Invariants
(Q², x, …) are computed from those same CSV components.

**Detector-style π± acceptance (optional kwargs on :func:`process_event_dataframe`):**
before choosing the leading target-hemisphere pion in Breit, discard charged pions unless
the lab η lies in the requested **proton-forward** interval (see
:func:`collider_forward_pion_pseudorapidity`: raw ``asinh(p_z/p_⊥)`` from native beam directions
is multiplied by ``DETECTOR_COLLIDER_PI_ETA_SIGN`` so bands match ``analyze_events_raw.EIC_REGIONS``).
Intermediate four-vector: ``flip_z(LT^{-1} P_h^{\mathrm{Breit}}, FLIP_Z_PTREL)``. For CSV-in-Breit
inputs this needs flipped-lab beam
four-vectors on the paired events row (``BEAM_FLIP_LAB_COLUMNS`` from
``generate_dis_editable_source_parquet.py``); see ``jet_hadron_observables_altered_parquet_filtered.py``.

Unchanged twin: split output puts altered events only under altered/. The driver loads the
pre-alter event from --full-event-csv (same event_id) when unchanged/event_XXXXXX.csv is absent.

Parallelism: ``--workers N`` (``spawn``) splits the altered CSV list across N processes; each
process builds **one** reused ``Pythia`` instance (same as serial). Default ``--workers 1``.

Requires: pythia8, pandas, numpy, matplotlib.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir


_ANALYSIS = Path(__file__).resolve().parent
if str(_ANALYSIS) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS))

from editable_source_parquet import BEAM_FLIP_LAB_COLUMNS  # noqa: E402

from jet_hadron_pi_pm_figures import write_split_pi_pm_comparison_pdfs  # noqa: E402

from diquark.analyze_events_raw import (  # noqa: E402
    FLIP_Z_PTREL,
    Qmax_ptrel,
    Qmin_ptrel,
    build_LT,
    dot4,
    flip_z,
    is_hadron,
    p3,
    xmax_ptrel,
    xmin_ptrel,
)

DEFAULT_SPLIT_ROOT = outputs_dir() / "dis_isr_parton_dataset" / "split_90_10"
DEFAULT_METADATA = outputs_dir() / "dis_isr_parton_dataset" / "dis_isr_event_metadata.csv"
DEFAULT_FULL_EVENT_CSV = outputs_dir() / "dis_isr_parton_dataset" / "dis_isr_full_event_record.csv"

STATUS_INJECT = 23
ZLC_MIN = 0.1
ZLC_MAX = 1.0 + 1e-6


def _load_jet_hadron_analyzer():
    """Same jet–hadron Breit helpers and plot binning as the shard pipeline."""
    path = PROJECT_ROOT / "scripts" / "analysis" / "analyze_jet_hadron_transverse_observables.py"
    spec = importlib.util.spec_from_file_location("_jet_hadron_transverse_obs_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


JH = _load_jet_hadron_analyzer()


def _load_find_struck():
    path = PROJECT_ROOT / "scripts" / "analysis" / "modify_dis_isr_parton_dataset.py"
    spec = importlib.util.spec_from_file_location("dis_modify", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.find_outgoing_struck_quark_noisr


find_outgoing_struck_quark_noisr = _load_find_struck()


def row_is_final_colored_qcd(row: pd.Series) -> bool:
    if int(row["isFinal"]) != 1:
        return False
    if int(row["status"]) <= 0:
        return False
    if int(row["daughter1"]) != 0 or int(row["daughter2"]) != 0:
        return False
    ap = abs(int(row["pdg_id"]))
    if ap == 21:
        return True
    if 1 <= ap <= 6:
        return True
    if 1000 <= ap < 10000 and (ap // 10) % 10 == 0:
        return True
    return False


def mask_final_colored_qcd(
    is_final: np.ndarray,
    status: np.ndarray,
    daughter1: np.ndarray,
    daughter2: np.ndarray,
    pdg_id: np.ndarray,
) -> np.ndarray:
    """Same boolean logic as ``row_is_final_colored_qcd`` (vectorized)."""
    is_final = np.asarray(is_final)
    status = np.asarray(status)
    d1 = np.asarray(daughter1)
    d2 = np.asarray(daughter2)
    pdg_id = np.asarray(pdg_id)
    m0 = (is_final == 1) & (status > 0) & (d1 == 0) & (d2 == 0)
    ap = np.abs(pdg_id.astype(np.int64, copy=False))
    gluon = ap == 21
    quark = (ap >= 1) & (ap <= 6)
    tens = (ap // 10) % 10
    excited = (ap >= 1000) & (ap < 10000) & (tens == 0)
    return m0 & (gluon | quark | excited)


def final_colored_partons(df: pd.DataFrame) -> pd.DataFrame:
    m = mask_final_colored_qcd(
        df["isFinal"].to_numpy(),
        df["status"].to_numpy(),
        df["daughter1"].to_numpy(),
        df["daughter2"].to_numpy(),
        df["pdg_id"].to_numpy(),
    )
    out = df.loc[m]
    pidx = out["particle_index"].to_numpy()
    if pidx.size <= 1 or np.all(pidx[1:] >= pidx[:-1]):
        return out.reset_index(drop=True)
    return out.sort_values("particle_index").reset_index(drop=True)


def color_balance_ok(df: pd.DataFrame) -> Tuple[bool, str]:
    tags: Dict[int, Tuple[int, int]] = {}
    for _, r in df.iterrows():
        for tag, is_col in [(int(r["col"]), True), (int(r["acol"]), False)]:
            if tag <= 0:
                continue
            nc, na = tags.get(tag, (0, 0))
            if is_col:
                nc += 1
            else:
                na += 1
            tags[tag] = (nc, na)
    for t, (nc, na) in sorted(tags.items()):
        if nc != 1 or na != 1:
            return False, f"tag {t}: n_col={nc} n_acol={na}"
    return True, "ok"


def row_p4(row: pd.Series) -> np.ndarray:
    return np.array(
        [float(row["E"]), float(row["px"]), float(row["py"]), float(row["pz"])],
        dtype=np.float64,
    )


def extract_beams_arrays(
    pdg: np.ndarray,
    status: np.ndarray,
    is_final: np.ndarray,
    E: np.ndarray,
    px: np.ndarray,
    py: np.ndarray,
    pz: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Same selection as ``extract_beams`` on a DataFrame: one pass over arrays.

    Incoming e: pdg==11, status<0, max pz. Scattered e: pdg==11, isFinal==1, max E.
    Proton: pdg==2212, status<0, max pz.
    """
    mask_in = (pdg == 11) & (status < 0)
    mask_out = (pdg == 11) & (is_final == 1)
    mask_p = (pdg == 2212) & (status < 0)

    pos_in = np.flatnonzero(mask_in)
    pos_out = np.flatnonzero(mask_out)
    pos_p = np.flatnonzero(mask_p)
    if pos_in.size == 0 or pos_out.size == 0 or pos_p.size == 0:
        return None, None, None, "beam_rows"

    i_ein = int(pos_in[int(np.argmax(pz[pos_in]))])
    i_esc = int(pos_out[int(np.argmax(E[pos_out]))])
    i_pin = int(pos_p[int(np.argmax(pz[pos_p]))])

    def pack(i: int) -> np.ndarray:
        return np.array([E[i], px[i], py[i], pz[i]], dtype=np.float64)

    return pack(i_ein), pack(i_esc), pack(i_pin), "ok"


def extract_beams(df: pd.DataFrame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], str]:
    """Match modify_dis_isr_parton_dataset.find_incoming_outgoing_electron (+ proton)."""
    return extract_beams_arrays(
        df["pdg_id"].to_numpy(),
        df["status"].to_numpy(),
        df["isFinal"].to_numpy(),
        df["E"].to_numpy(dtype=np.float64),
        df["px"].to_numpy(dtype=np.float64),
        df["py"].to_numpy(dtype=np.float64),
        df["pz"].to_numpy(dtype=np.float64),
    )


def flipped_lab_beam_arrays_from_event_row(
    row: pd.Series,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Reconstruct flipped-lab beam four-vectors from Parquet ``BEAM_FLIP_LAB_COLUMNS``."""
    idx = row.index
    for c in BEAM_FLIP_LAB_COLUMNS:
        if c not in idx:
            return None
        if pd.isna(row[c]):
            return None
    ein_ev = np.array(
        [
            float(row["ein_fl_E"]),
            float(row["ein_fl_px"]),
            float(row["ein_fl_py"]),
            float(row["ein_fl_pz"]),
        ],
        dtype=np.float64,
    )
    esc_ev = np.array(
        [
            float(row["esc_fl_E"]),
            float(row["esc_fl_px"]),
            float(row["esc_fl_py"]),
            float(row["esc_fl_pz"]),
        ],
        dtype=np.float64,
    )
    pin_ev = np.array(
        [
            float(row["pin_fl_E"]),
            float(row["pin_fl_px"]),
            float(row["pin_fl_py"]),
            float(row["pin_fl_pz"]),
        ],
        dtype=np.float64,
    )
    return ein_ev, esc_ev, pin_ev


def dis_xy_q_within_selection(
    x: float,
    Q: float,
    *,
    x_exclusive_floor: Optional[float] = None,
    q_open_interval_gev: Optional[Tuple[float, float]] = None,
) -> bool:
    """
    Bjorken x and virtuality Q cuts for DIS reinjection jets.

    When ``x_exclusive_floor`` is set, require ``x > floor`` (strict) with ``x <= 1.0``.
    When unset, keep ``xmin_ptrel <= x <= xmax_ptrel``.

    When ``q_open_interval_gev`` is ``(q_lo, q_hi)``, require strict ``q_lo < Q < q_hi``.
    When unset, keep ``Qmin_ptrel <= Q <= Qmax_ptrel`` from shard defaults.
    """
    if Q <= 0 or not math.isfinite(Q) or not math.isfinite(x):
        return False
    if x_exclusive_floor is None:
        if not (xmin_ptrel <= x <= xmax_ptrel):
            return False
    else:
        if not (float(x_exclusive_floor) < x <= 1.0):
            return False

    if q_open_interval_gev is None:
        if not (Qmin_ptrel <= Q <= Qmax_ptrel):
            return False
    else:
        q_lo, q_hi = float(q_open_interval_gev[0]), float(q_open_interval_gev[1])
        if not (q_lo < Q < q_hi):
            return False
    return True


def kinematics_lab_to_breit_lt_from_flip_lab_beams(
    ein_ev: np.ndarray,
    esc_ev: np.ndarray,
    pin_ev: np.ndarray,
    *,
    x_exclusive_floor: Optional[float] = None,
    q_open_interval_gev: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Same lab→Breit ``LT`` as the reinjection path when momenta live in flipped lab frame.
    Mirrors ``process_event_dataframe`` after ``flip_z(..., FLIP_Z_PTREL)``.
    """
    ein_ev = np.asarray(ein_ev, dtype=np.float64)
    esc_ev = np.asarray(esc_ev, dtype=np.float64)
    pin_ev = np.asarray(pin_ev, dtype=np.float64)
    qmu = ein_ev - esc_ev
    Q2 = -(qmu[0] * qmu[0] - qmu[1] * qmu[1] - qmu[2] * qmu[2] - qmu[3] * qmu[3])
    if Q2 <= 0:
        return None, "Q2_eta"
    Q = float(np.sqrt(Q2))
    qT = float(np.hypot(qmu[1], qmu[2]))
    p_dot_q = float(
        pin_ev[0] * qmu[0] - pin_ev[1] * qmu[1] - pin_ev[2] * qmu[2] - pin_ev[3] * qmu[3]
    )
    if p_dot_q == 0:
        return None, "pdotq_eta"
    x = Q2 / (2.0 * p_dot_q)
    if not dis_xy_q_within_selection(
        x,
        Q,
        x_exclusive_floor=x_exclusive_floor,
        q_open_interval_gev=q_open_interval_gev,
    ):
        return None, "xQ_window_eta"
    Ee = float(ein_ev[0])
    Ep = float(pin_ev[0])
    S = 4.0 * Ee * Ep
    y = Q2 / (S * x)
    phiq = float(np.arctan2(qmu[2], qmu[1]))
    LT = build_LT(Ee, Ep, qmu, x, y, qT, phiq, S)
    if LT is None:
        return None, "build_LT_eta"
    return LT, ""


def hadron_pseudorapidity(p4: np.ndarray) -> float:
    """Standard η(p) from (E,px,py,pz) via θ: η = asinh(pz / p⊥)."""
    px = float(p4[1])
    py = float(p4[2])
    pz = float(p4[3])
    pt = math.hypot(px, py)
    if pt <= 0:
        raise ValueError("zero transverse momentum")
    return float(math.asinh(pz / pt))


# Raw PYTHIA Beams:idA=11,idB=2212 η(p) can differ by a sign from proton-forward detector η
# (``EIC_REGIONS`` in analyze_events_raw.py). Keep filter + ``pion_lab_eta`` consistent.
DETECTOR_COLLIDER_PI_ETA_SIGN = -1.0


def collider_forward_pion_pseudorapidity(p_beam_lab: np.ndarray) -> float:
    """
    Hadron η used for EIC-style acceptance on ``p_beam_lab`` (after LT^{-1}+flip_z undo path).
    Applies :data:`DETECTOR_COLLIDER_PI_ETA_SIGN` to :func:`hadron_pseudorapidity`.
    """
    raw = float(hadron_pseudorapidity(p_beam_lab))
    return float(DETECTOR_COLLIDER_PI_ETA_SIGN * raw)


def pion_lab_eta_native_beam_from_breit(
    p4_breit: np.ndarray,
    *,
    csv_momenta_breit: bool,
    event_row: Optional[pd.Series],
    LT_breit_lab_to_process: np.ndarray,
    x_exclusive_floor: Optional[float],
    q_open_interval_gev: Optional[Tuple[float, float]],
) -> Optional[float]:
    """
    Proton-forward lab pseudorapidity of the π: same LT^{-1}+``flip_z`` undo path as
    ``discard_charged_pions_outside_collider_eta``, then :func:`collider_forward_pion_pseudorapidity`.

    ``LT`` is ``LT_eta`` rebuilt from flipped-lab beams for Breit-CSV parquet rows; otherwise
    the lab→Breit matrix used elsewhere in reinjection for lab CSV inputs.
    """
    if csv_momenta_breit:
        if event_row is None:
            return None
        blt = flipped_lab_beam_arrays_from_event_row(event_row)
        if blt is None:
            return None
        LT_eta, _err = kinematics_lab_to_breit_lt_from_flip_lab_beams(
            *blt,
            x_exclusive_floor=x_exclusive_floor,
            q_open_interval_gev=q_open_interval_gev,
        )
        if LT_eta is None:
            return None
    else:
        LT_eta = np.asarray(LT_breit_lab_to_process, dtype=np.float64)

    try:
        LT_inv = np.linalg.inv(LT_eta)
    except np.linalg.LinAlgError:
        LT_inv = np.linalg.pinv(LT_eta)
    p_flipped_lab = LT_inv @ np.asarray(p4_breit, dtype=np.float64)
    p_beam_lab = flip_z(p_flipped_lab, FLIP_Z_PTREL)
    try:
        return collider_forward_pion_pseudorapidity(p_beam_lab)
    except ValueError:
        return None


def discard_charged_pions_outside_collider_eta(
    hadrons: List[Tuple[int, np.ndarray]],
    LT_lab_to_breit: np.ndarray,
    eta_min: float,
    eta_max: float,
) -> List[Tuple[int, np.ndarray]]:
    """
    Drop |pdg|==211 charged pions unless ``eta_min < η < η_max`` with **proton-forward** η
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
        if ap == 211 and is_hadron(int(pid)):
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


def resolve_struck_index(
    df: pd.DataFrame,
    event_id: int,
    meta_path: Optional[Path],
    md_map: Optional[pd.DataFrame],
) -> Tuple[Optional[int], str]:
    if meta_path is not None and meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return int(meta["struck_quark_index"]), "meta_json"
    if md_map is not None and event_id in md_map.index:
        row = md_map.loc[event_id]
        so = int(row["struck_outgoing_index"])
        if so > 0:
            return so, "metadata_outgoing"
        si = int(row["struck_incoming_index"])
        reco = find_outgoing_struck_quark_noisr(df, si)
        if reco["success"]:
            return int(reco["selected_row"]["particle_index"]), "reco"
    return None, "unresolved"


def parton_p4_breit_for_inject(
    row: pd.Series, LT: np.ndarray, csv_momenta_breit: bool
) -> Tuple[float, float, float, float, float]:
    """Return (E, px, py, pz, m_on_shell) for PYTHIA append: lab→Breit or already-Breit CSV."""
    E = float(row["E"])
    px, py, pz = float(row["px"]), float(row["py"]), float(row["pz"])
    if csv_momenta_breit:
        m2 = E * E - px * px - py * py - pz * pz
        m = float(np.sqrt(max(0.0, m2)))
        return E, px, py, pz, m
    v = np.array([E, px, py, pz], dtype=np.float64)
    v = flip_z(v, FLIP_Z_PTREL)
    vb = LT @ v
    E, px, py, pz = float(vb[0]), float(vb[1]), float(vb[2]), float(vb[3])
    m2 = E * E - px * px - py * py - pz * pz
    m = float(np.sqrt(max(0.0, m2)))
    return E, px, py, pz, m


def build_pythia_reinjector(
    *,
    e_beam_gev: float = 18.0,
    p_beam_gev: float = 275.0,
    parton_level_isr: bool = True,
    parton_level_fsr: bool = True,
    random_seed: Optional[int] = None,
) -> pythia8.Pythia:
    """
    Single long-lived PYTHIA instance for repeated reinjection (reset event between calls).

    Nominal beams should match editable / CSV generation settings (defaults 18×275 GeV e–p,
    ``frameType = 2``) so PDF and ISR see the same collision system even though
    ``ProcessLevel:all = off`` (the hard system is injected via ``event.append``).
    ``PDF:lepton = off`` matches that generator style.
    ``parton_level_isr`` / ``parton_level_fsr`` map to ``PartonLevel:ISR`` / ``PartonLevel:FSR``.
    If ``random_seed`` is set, ``Random:seed`` / ``Random:setSeed`` are applied before ``init()`` for
    reproducibility.
    """
    def _flag(x: bool) -> str:
        return "on" if x else "off"

    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eA = {float(e_beam_gev)}")
    p.readString(f"Beams:eB = {float(p_beam_gev)}")
    p.readString("Beams:frameType = 2")
    p.readString("PDF:lepton = off")
    p.readString("ProcessLevel:all = off")
    p.readString(f"PartonLevel:ISR = {_flag(parton_level_isr)}")
    p.readString(f"PartonLevel:FSR = {_flag(parton_level_fsr)}")
    p.readString("PartonLevel:MPI = off")
    p.readString("HadronLevel:all = on")
    p.readString("Print:quiet = on")
    if random_seed is not None:
        p.readString(f"Random:seed = {int(random_seed)}")
        p.readString("Random:setSeed = on")
    if not p.init():
        raise RuntimeError("PYTHIA init failed in build_pythia_reinjector")
    return p


def run_pythia_reinject_collect(
    p: pythia8.Pythia,
    partons: pd.DataFrame,
    LT: np.ndarray,
    csv_momenta_breit: bool,
    phase_times: Optional[MutableMapping[str, float]] = None,
) -> Tuple[bool, str, List[Tuple[int, np.ndarray]]]:
    """
    Append colored partons (Breit-frame momenta for showering), one pythia.next().
    Reuses ``p`` — call ``build_pythia_reinjector()`` once per process.

    If ``phase_times`` is provided, add wall-time seconds for keys
    ``reinject_prep`` (Python parton four-vectors before append),
    ``pythia_append`` (``event.reset()`` and ``event.append`` only),
    ``pythia_next``, and ``hadron_scan`` (no physics change).
    """
    from time import perf_counter

    t_append0 = perf_counter()
    p.event.reset()
    if phase_times is not None:
        phase_times["pythia_append"] = phase_times.get("pythia_append", 0.0) + (
            perf_counter() - t_append0
        )
    try:
        for _, r in partons.iterrows():
            t_prep0 = perf_counter()
            E, px, py, pz, m = parton_p4_breit_for_inject(r, LT, csv_momenta_breit)
            if phase_times is not None:
                phase_times["reinject_prep"] = phase_times.get("reinject_prep", 0.0) + (
                    perf_counter() - t_prep0
                )
            t_ap0 = perf_counter()
            p.event.append(
                int(r["pdg_id"]),
                STATUS_INJECT,
                int(r["col"]),
                int(r["acol"]),
                px,
                py,
                pz,
                E,
                m,
            )
            if phase_times is not None:
                phase_times["pythia_append"] = phase_times.get("pythia_append", 0.0) + (
                    perf_counter() - t_ap0
                )
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}", []

    t_next0 = perf_counter()
    try:
        ok = bool(p.next())
    except Exception as exc:
        if phase_times is not None:
            phase_times["pythia_next"] = phase_times.get("pythia_next", 0.0) + (
                perf_counter() - t_next0
            )
        return False, f"{type(exc).__name__}: {exc}", []

    if phase_times is not None:
        phase_times["pythia_next"] = phase_times.get("pythia_next", 0.0) + (
            perf_counter() - t_next0
        )

    if not ok:
        return False, "pythia.next() failed", []

    t_scan0 = perf_counter()
    out: List[Tuple[int, np.ndarray]] = []
    for i in range(p.event.size()):
        pp = p.event[i]
        if not pp.isFinal() or not pp.isHadron():
            continue
        pid = int(pp.id())
        v = np.array([pp.e(), pp.px(), pp.py(), pp.pz()], dtype=np.float64)
        out.append((pid, v))
    if phase_times is not None:
        phase_times["hadron_scan"] = phase_times.get("hadron_scan", 0.0) + (
            perf_counter() - t_scan0
        )
    return True, "", out


def run_pythia_collect_final_hadrons(
    partons: pd.DataFrame,
    LT: np.ndarray,
    csv_momenta_breit: bool,
    *,
    e_beam_gev: float = 18.0,
    p_beam_gev: float = 275.0,
    parton_level_isr: bool = True,
    parton_level_fsr: bool = True,
) -> Tuple[bool, str, List[Tuple[int, np.ndarray]]]:
    """
    Append colored partons in the Breit-frame momenta PYTHIA should use for showering.
    Final hadrons are read back in that same frame — do not Breit-boost them again.
    """
    p = build_pythia_reinjector(
        e_beam_gev=e_beam_gev,
        p_beam_gev=p_beam_gev,
        parton_level_isr=parton_level_isr,
        parton_level_fsr=parton_level_fsr,
    )
    return run_pythia_reinject_collect(p, partons, LT, csv_momenta_breit)


def best_leading_charged_pion_breit(
    hadrons: List[Tuple[int, np.ndarray]],
    *,
    breit_hemisphere: str = "target",
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    """
    Single leading charged π in a Breit hemisphere (post-injection hadron list):

    ``target``: max E among |pdg|==211 with ``pz_Breit > 0`` (proton / target-fragmentation side).
    ``current``: max E among |pdg|==211 with ``pz_Breit < 0`` (current hemisphere).

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
        if ap != 211 or not is_hadron(int(pid)):
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


def compute_observables_for_pion(
    pion_breit: np.ndarray,
    pT_jet: np.ndarray,
    p_proton_breit: np.ndarray,
    q_breit: np.ndarray,
) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], str]:
    den = dot4(p_proton_breit, q_breit)
    if den <= 0:
        return None, None, None, None, None, "zLC_den"
    zLC = dot4(pion_breit, q_breit) / den
    if zLC < ZLC_MIN or zLC > ZLC_MAX:
        return None, None, None, None, float(zLC), "zLC_range"

    pT_h = JH._pT_vec_breit(pion_breit)
    phi_hj = JH._angle_between_2d(pT_jet, pT_h)
    diff_mag = float(np.linalg.norm(pT_jet - pT_h))
    sum_mag = float(np.linalg.norm(pT_jet + pT_h))
    pbar_t = 0.5 * sum_mag
    return phi_hj, diff_mag, pbar_t, sum_mag, float(zLC), "ok"


def process_event_dataframe(
    df: pd.DataFrame,
    struck_idx: int,
    sample: str,
    csv_label: str,
    csv_momenta_breit: bool = False,
    pythia_reuse: Optional[pythia8.Pythia] = None,
    *,
    event_row: Optional[pd.Series] = None,
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
    """Return exactly one row per event (leading |pdg|==211 in chosen Breit hemisphere or failure).

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

    if pythia_reuse is not None:
        py_ok, py_err, hadrons = run_pythia_reinject_collect(
            pythia_reuse, partons, LT, csv_momenta_breit
        )
    else:
        py_ok, py_err, hadrons = run_pythia_collect_final_hadrons(
            partons,
            LT,
            csv_momenta_breit,
            e_beam_gev=pythia_e_beam_gev,
            p_beam_gev=pythia_p_beam_gev,
            parton_level_isr=pythia_parton_level_isr,
            parton_level_fsr=pythia_parton_level_fsr,
        )
    if not py_ok:
        return _fail("pythia", pythia_error=py_err)

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
        hadrons = discard_charged_pions_outside_collider_eta(
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

    win_pid, pib = best_leading_charged_pion_breit(
        hadrons, breit_hemisphere=pion_breit_hemisphere
    )
    pion_tag = "piplus" if win_pid == 211 else "piminus" if win_pid == -211 else ""
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


def process_event_csv(
    csv_path: Path,
    struck_idx: int,
    sample: str,
    csv_momenta_breit: bool = False,
    pythia_reuse: Optional[pythia8.Pythia] = None,
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path).sort_values("particle_index")
    return process_event_dataframe(
        df,
        struck_idx,
        sample,
        csv_path.name,
        csv_momenta_breit,
        pythia_reuse=pythia_reuse,
        **kwargs,
    )


def collect_pi_pm_records(
    csv_files: List[Path],
    *,
    split_root: Path,
    unchanged_dir: Path,
    md_map: Optional[pd.DataFrame],
    full_ev: Optional[pd.DataFrame],
    csv_momenta_breit: bool,
    pythia_reuse: Optional[pythia8.Pythia] = None,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Process a list of altered ``event_*.csv`` paths; return (records, n_matched_pairs).
    ``pythia_reuse``: one long-lived instance per process (required for sane throughput).
    """
    altered_dir = split_root / "altered"
    all_records: List[Dict[str, Any]] = []
    n_matched_pairs = 0

    for alt_path in csv_files:
        stem = alt_path.stem
        un_path = unchanged_dir / f"{stem}.csv"
        try:
            event_id = int(stem.split("_")[1])
        except (IndexError, ValueError):
            continue

        meta_path = altered_dir / f"{stem}.meta.json"
        df_alt = pd.read_csv(alt_path).sort_values("particle_index").reset_index(drop=True)
        ev0 = int(df_alt["event_id"].iloc[0])

        if un_path.is_file():
            df_un = pd.read_csv(un_path).sort_values("particle_index").reset_index(drop=True)
            un_label = un_path.name
        elif full_ev is not None:
            df_un = full_ev[full_ev["event_id"] == event_id].copy()
            if len(df_un) == 0:
                continue
            df_un = df_un.sort_values("particle_index").reset_index(drop=True)
            un_label = f"{stem}.csv@full_event_record"
        else:
            continue

        n_matched_pairs += 1

        idx_a, _ = resolve_struck_index(df_alt, ev0, meta_path, md_map)
        ev1 = int(df_un["event_id"].iloc[0])
        idx_u, _ = resolve_struck_index(df_un, ev1, None, md_map)

        if idx_a is None:
            for sample in ("altered", "unchanged"):
                all_records.append(
                    {
                        "event_id": ev0,
                        "sample": sample,
                        "pion": "",
                        "pdg_id": 0,
                        "ok": False,
                        "reason": "struck_unresolved",
                        "csv": (alt_path.name if sample == "altered" else un_label),
                    }
                )
            continue

        if idx_u is None:
            all_records.append(
                {
                    "event_id": ev0,
                    "sample": "unchanged",
                    "pion": "",
                    "pdg_id": 0,
                    "ok": False,
                    "reason": "struck_unresolved_unchanged",
                    "csv": un_label,
                }
            )
            all_records.extend(
                process_event_dataframe(
                    df_alt, idx_a, "altered", alt_path.name, csv_momenta_breit, pythia_reuse=pythia_reuse
                )
            )
            continue

        for sample, dff, struck, label in (
            ("altered", df_alt, idx_a, alt_path.name),
            ("unchanged", df_un, idx_u, un_label),
        ):
            all_records.extend(
                process_event_dataframe(dff, struck, sample, label, csv_momenta_breit, pythia_reuse=pythia_reuse)
            )

    return all_records, n_matched_pairs


def _worker_pi_pm_chunk(payload: Tuple[List[str], Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Child process: rebuild paths, load metadata slice, one PYTHIA instance."""
    path_strs, cfg = payload
    csv_files = [Path(p) for p in path_strs]
    if not csv_files:
        return [], 0
    split_root = Path(cfg["split_root"])
    unchanged_dir = split_root / "unchanged"
    md_map: Optional[pd.DataFrame] = None
    mpath = cfg.get("metadata_csv")
    if mpath:
        md = pd.read_csv(Path(mpath)).sort_values("event_id").reset_index(drop=True)
        md_map = md.set_index("event_id")
    full_ev: Optional[pd.DataFrame] = None
    fpath = cfg.get("full_event_csv")
    if fpath and cfg.get("needed_ids"):
        fe = pd.read_csv(Path(fpath))
        ids = cfg["needed_ids"]
        full_ev = fe[fe["event_id"].isin(ids)].sort_values(["event_id", "particle_index"])
    p = build_pythia_reinjector()
    return collect_pi_pm_records(
        csv_files,
        split_root=split_root,
        unchanged_dir=unchanged_dir,
        md_map=md_map,
        full_ev=full_ev,
        csv_momenta_breit=bool(cfg["csv_momenta_breit"]),
        pythia_reuse=p,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Jet–hadron observables for split CSV reinjection (Breit, jet pT from k_out): "
            "one row per event — leading charged π in the target hemisphere."
        )
    )
    ap.add_argument("--split-root", type=Path, default=DEFAULT_SPLIT_ROOT)
    ap.add_argument("--metadata-csv", type=Path, default=DEFAULT_METADATA)
    ap.add_argument(
        "--full-event-csv",
        type=Path,
        default=DEFAULT_FULL_EVENT_CSV,
        help="Pre-split full event table; used to load unchanged twin when not in unchanged/.",
    )
    ap.add_argument("--max-events", type=int, default=0, help="If >0, cap altered events processed.")
    ap.add_argument(
        "--csv-momenta-frame",
        choices=("lab", "breit"),
        default="lab",
        help="lab: CSV is lab frame (apply flip_z + build_LT). breit: CSV from generate_dis_isr_parton_dataset Breit output (no second LT).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Defaults to split_root/jet_hadron_pi_pm_observables.csv",
    )
    ap.add_argument(
        "--figure-dir",
        type=Path,
        default=None,
        help="Directory for comparison PDFs (default: split-root); see jet_hadron_pi_pm_figures.py.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (spawn). Each worker reuses one PYTHIA instance. Default 1 (serial).",
    )
    args = ap.parse_args()
    csv_momenta_breit = args.csv_momenta_frame == "breit"

    altered_dir = args.split_root / "altered"
    unchanged_dir = args.split_root / "unchanged"
    if not altered_dir.is_dir():
        raise SystemExit(f"altered dir not found: {altered_dir}")
    if not unchanged_dir.is_dir():
        raise SystemExit(f"unchanged dir not found: {unchanged_dir}")

    md_map: Optional[pd.DataFrame] = None
    if args.metadata_csv.is_file():
        md = pd.read_csv(args.metadata_csv).sort_values("event_id").reset_index(drop=True)
        md_map = md.set_index("event_id")

    csv_files = sorted(altered_dir.glob("event_*.csv"))
    if args.max_events > 0:
        csv_files = csv_files[: args.max_events]

    needed_ids: List[int] = []
    for p in csv_files:
        try:
            needed_ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass

    full_ev: Optional[pd.DataFrame] = None
    if args.full_event_csv.is_file() and needed_ids:
        fe = pd.read_csv(args.full_event_csv)
        full_ev = fe[fe["event_id"].isin(needed_ids)].sort_values(["event_id", "particle_index"])

    n_altered_scanned = len(csv_files)
    n_workers = max(1, int(args.workers))

    if n_workers <= 1:
        p_main = build_pythia_reinjector()
        all_records, n_matched_pairs = collect_pi_pm_records(
            csv_files,
            split_root=args.split_root,
            unchanged_dir=unchanged_dir,
            md_map=md_map,
            full_ev=full_ev,
            csv_momenta_breit=csv_momenta_breit,
            pythia_reuse=p_main,
        )
    else:
        arr = np.empty(len(csv_files), dtype=object)
        for i, p in enumerate(csv_files):
            arr[i] = p
        chunks = [list(x) for x in np.array_split(arr, n_workers)]
        payloads: List[Tuple[List[str], Dict[str, Any]]] = []
        for chunk_paths in chunks:
            if not chunk_paths:
                continue
            chunk_needed: List[int] = []
            for p in chunk_paths:
                try:
                    chunk_needed.append(int(Path(p).stem.split("_")[1]))
                except (IndexError, ValueError):
                    pass
            cfg = {
                "split_root": str(args.split_root.resolve()),
                "metadata_csv": str(args.metadata_csv.resolve()) if args.metadata_csv.is_file() else "",
                "full_event_csv": str(args.full_event_csv.resolve()) if args.full_event_csv.is_file() else "",
                "needed_ids": chunk_needed,
                "csv_momenta_breit": csv_momenta_breit,
            }
            payloads.append(([str(p) for p in chunk_paths], cfg))
        ctx = mp.get_context("spawn")
        all_records = []
        n_matched_pairs = 0
        with ctx.Pool(processes=min(n_workers, len(payloads))) as pool:
            for recs, nm in pool.map(_worker_pi_pm_chunk, payloads):
                all_records.extend(recs)
                n_matched_pairs += nm

    out_csv = args.out_csv or (args.split_root / "jet_hadron_pi_pm_observables.csv")
    figure_dir = args.figure_dir or args.split_root

    pd.DataFrame(all_records).to_csv(out_csv, index=False)
    pdfs = write_split_pi_pm_comparison_pdfs(all_records, figure_dir)

    n_ok = sum(1 for r in all_records if r.get("ok"))
    print(f"Wrote {out_csv} ({len(all_records)} rows, {n_ok} ok)")
    for p in pdfs:
        print(f"Wrote {p}")

    # --- compact run summary ---
    def _truthy_ok(r: Dict[str, Any]) -> bool:
        v = r.get("ok")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)

    n_pythia_ok_rows = sum(1 for r in all_records if r.get("pythia_ok") is True)
    n_reinject_paths = n_pythia_ok_rows

    n_pi_plus = sum(
        1 for r in all_records if r.get("pion") == "piplus" and _truthy_ok(r)
    )
    n_pi_minus = sum(
        1 for r in all_records if r.get("pion") == "piminus" and _truthy_ok(r)
    )
    reasons = Counter(str(r.get("reason", "")) for r in all_records if not _truthy_ok(r))

    print()
    print("=== jet_hadron_observables_split_pi_pm summary ===")
    print(f"  altered CSV files scanned (--max-events cap):     {n_altered_scanned}")
    print(f"  matched altered/unchanged pairs processed:        {n_matched_pairs}")
    print(f"  table rows written:                               {len(all_records)}")
    print(f"  rows with ok=True (valid observable):             {n_ok}")
    print(f"  rows with pythia_ok=True:                         {n_pythia_ok_rows}")
    print(f"  reinject paths with pythia success (one row each):  {n_reinject_paths}")
    print(f"  valid rows with leading π+ (ok, pion=piplus):       {n_pi_plus}")
    print(f"  valid rows with leading π− (ok, pion=piminus):      {n_pi_minus}")
    print("  skip/failure reasons (not ok), top counts:")
    for reason, cnt in reasons.most_common(12):
        if reason:
            print(f"    {reason}: {cnt}")
    print("=" * 50)


if __name__ == "__main__":
    main()

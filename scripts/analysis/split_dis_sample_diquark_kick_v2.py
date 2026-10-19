#!/usr/bin/env python3
"""
Split DIS parton-level CSV events into 90% unchanged vs 10% altered (v2 surgery).

**v2 physics** (companion to ``split_dis_sample_diquark_kick.py`` v1):

Altered events: replace final (ud) diquark (PDG 2101) with a 3-body split
``(ud) -> [d] + (us) + sbar`` (PYTHIA ``su_0`` diquark PDG 3201), then apply a
balanced px kick to the stripped [d] and the outgoing struck quark.

**Longitudinal partition (v2):** daughter momenta use proton-momentum fractions
parameterized by the event Bjorken ``x_struck`` (events table column ``x``), not
fixed fractions of the (ud) diquark four-momentum:

    x_d (stripped)  = x_struck
    x_sbar          = 0.01
    x_us (daughter) = 0.99 - 2 * x_struck

Events with ``x_struck > 0.495`` are rejected (daughter share non-positive).
Transverse (px, py) of the three daughters sum to the original diquark (px, py);
longitudinal pz is set from ``f_i * p_proton_breit_pz`` and no longer matches the
diquark pz sum.

**Kick (v2):** magnitude ``delta_px * sin(theta)`` with ``theta ~ Uniform[0, pi]``,
applied along Breit x (same direction as v1; magnitude-only change). θ is drawn from
``Random(f"{seed_for_kick}:{event_id}")`` per event so the same ``--seed`` and ``event_id``
yield the same kick in serial and parallel runs.

We want event-by-event comparisons between kicked and unkicked events after PYTHIA
reinjects and hadronizes them; this script prepares paired hard-event CSVs (unchanged
vs altered) from the same source sample.

No junctions, no LHE, no full history replay.

Companion tools: print_split_90_10_summary.py, inspect_altered_event.py, reinject_altered_events.py

**Parquet (preferred for large samples):** ``split_dis_sample_diquark_kick_parquet.py`` reads
``editable_source_v1`` shards and writes ``split_90_10_parquet/`` (see ``split_output_parquet.py``).
This CSV driver remains for legacy workflows; production Parquet alteration uses
``transform_editable_all_altered_parquet.py --surgery-version v2``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.paths import count_files_under, outputs_dir, write_run_manifest

OUTDIR = outputs_dir() / "dis_isr_parton_dataset"
FULL_EVENT_CSV = OUTDIR / "dis_isr_full_event_record.csv"
METADATA_CSV = OUTDIR / "dis_isr_event_metadata.csv"

# Output root: two subdirs unchanged/ and altered/
DEFAULT_SPLIT_ROOT = OUTDIR / "split_90_10"

M_LIGHT = 0.33
DIQUARK_UD0 = 2101
# (u,s) diquark in PYTHIA naming order (``su_0``); same flavor content as "(us)".
DIQUARK_SU0 = 3201
TOL_3M = 1e-6
TOL_ONSHELL = 1e-3


def _load_dis_helpers():
    path = Path(__file__).resolve().parent / "modify_dis_isr_parton_dataset.py"
    spec = importlib.util.spec_from_file_location("dis_modify", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_DIS = _load_dis_helpers()
find_outgoing_struck_quark_noisr = _DIS.find_outgoing_struck_quark_noisr


_pythia_pd = None


def _pythia_mass0(pid: int) -> float:
    global _pythia_pd
    try:
        import pythia8
    except ImportError:
        ap = abs(pid)
        return {2101: 0.57933, 3201: 0.80473, 1103: 0.57933}.get(ap, M_LIGHT)
    if _pythia_pd is None:
        p = pythia8.Pythia()
        p.readString("Print:quiet = on")
        p.init()
        _pythia_pd = p.particleData
    return float(_pythia_pd.m0(abs(pid)))


def mass_for_pid(pid: int) -> float:
    ap = abs(pid)
    if ap in (1, 2, 3):
        return M_LIGHT
    if 1000 <= ap < 10000:
        return _pythia_mass0(pid)
    return M_LIGHT


def e_on_shell(px: float, py: float, pz: float, m: float) -> float:
    return math.sqrt(max(0.0, px * px + py * py + pz * pz + m * m))


def kinematic_extras(px: float, py: float, pz: float, E: float) -> Tuple[float, float, float, float]:
    pT = math.hypot(px, py)
    if pT < 1e-14:
        eta = math.copysign(1e6, pz) if pz != 0 else 0.0
    else:
        eta = math.asinh(pz / pT)
    phi = math.atan2(py, px)
    m = math.sqrt(max(0.0, E * E - px * px - py * py - pz * pz))
    return pT, eta, phi, m


def find_final_diquark_kick_partner_row(df_event: pd.DataFrame) -> Optional[pd.Series]:
    """Prefer final ud0 (2101); else highest-E final diquark-like (1000 <= |pdg| < 10000)."""
    r0 = find_final_ud0_diquark(df_event)
    if r0 is not None:
        return r0
    cand = df_event[(df_event["isFinal"] == 1) & (df_event["pdg_id"].abs() >= 1000) & (df_event["pdg_id"].abs() < 10000)]
    if cand.empty:
        return None
    return cand.loc[cand["E"].idxmax()]


def find_final_ud0_diquark(df_event: pd.DataFrame) -> Optional[pd.Series]:
    cand = df_event[(df_event["isFinal"] == 1) & (df_event["pdg_id"] == DIQUARK_UD0)]
    if cand.empty:
        return None
    if len(cand) == 1:
        return cand.iloc[0]
    idxs = set(int(x) for x in cand["particle_index"].values)
    if 32 in idxs:
        return cand[cand["particle_index"] == 32].iloc[0]
    return cand.loc[cand["E"].idxmax()]


def next_free_color_tag(df_event: pd.DataFrame) -> int:
    used = set()
    for _, r in df_event.iterrows():
        c, a = int(r["col"]), int(r["acol"])
        if c > 0:
            used.add(c)
        if a > 0:
            used.add(a)
    return (max(used) + 1) if used else 501


def validate_surgery_color_tags(df: pd.DataFrame, tag_a: int, tag_b: int) -> Tuple[bool, str]:
    """
    Full DIS CSVs often have multiple lines with the same col tag (e.g. several col=101).
    We require the *new* string A to appear exactly once as col and once as acol.
    For B we only check that at least one particle still carries acol=B (the created
    antiquark), matching the moved diquark endpoint.
    """
    nc_a = sum(int(r["col"]) == tag_a for _, r in df.iterrows())
    na_a = sum(int(r["acol"]) == tag_a for _, r in df.iterrows())
    if nc_a != 1 or na_a != 1:
        return False, f"tag A ({tag_a}): n_col={nc_a} n_acol={na_a}"
    na_b = sum(int(r["acol"]) == tag_b for _, r in df.iterrows())
    if na_b < 1:
        return False, f"tag B ({tag_b}): missing acol endpoint"
    return True, "ok"


def channel_ab(ch: str) -> Tuple[int, int, int]:
    """(stripped_id, daughter_diquark_id, antiquark_id)."""
    if ch == "C":
        return 1, DIQUARK_SU0, -3  # (ud) -> d + (us-like su_0) + sbar
    raise ValueError(f"unknown split channel {ch!r} (expected 'C')")


def build_three_body_from_diquark(
    dq_row: pd.Series,
    channel: str,
    x_struck: float,
    p_proton_breit_pz: float,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    v2 partition: longitudinal fractions are of the proton, not the diquark.
        x_d (stripped)    = x_struck
        x_sbar            = 0.01
        x_us (daughter)   = 0.99 - 2 * x_struck

    Caller must reject events with x_struck > 0.495 before calling this
    (the (us) share would go to <= 0).

    Transverse (px, py) of the three daughters: assigned proportionally to the
    original diquark transverse momentum, weighted by the longitudinal fractions
    above (so that the sum reproduces the diquark's (px, py) exactly — same
    momentum-conservation guarantee as v1).
    """
    if x_struck > 0.495:
        raise ValueError(f"x_struck={x_struck:.4f} > 0.495; caller must reject before this point")

    f_d = float(x_struck)
    f_sbar = 0.01
    f_us = 0.99 - 2.0 * float(x_struck)
    denom = f_d + f_sbar + f_us  # = 1 - x_struck

    # Longitudinal pz: distributed in proton-fraction units, scaled to Breit pz of proton.
    pz_d = f_d * float(p_proton_breit_pz)
    pz_sbar = f_sbar * float(p_proton_breit_pz)
    pz_us = f_us * float(p_proton_breit_pz)

    # Transverse (px, py): proportional to the longitudinal weights so the three-body
    # sum reproduces the original diquark's transverse momentum exactly.
    px_dq = float(dq_row["px"])
    py_dq = float(dq_row["py"])
    px_d = (f_d / denom) * px_dq
    py_d = (f_d / denom) * py_dq
    px_sbar = (f_sbar / denom) * px_dq
    py_sbar = (f_sbar / denom) * py_dq
    px_us = (f_us / denom) * px_dq
    py_us = (f_us / denom) * py_dq

    sid, did, aid = channel_ab(channel)
    ms, mdq, ma = mass_for_pid(sid), mass_for_pid(did), mass_for_pid(aid)

    Es = e_on_shell(px_d, py_d, pz_d, ms)
    Ea = e_on_shell(px_sbar, py_sbar, pz_sbar, ma)
    Ed = e_on_shell(px_us, py_us, pz_us, mdq)

    return (
        _row_kin(sid, px_d, py_d, pz_d, Es, ms),
        _row_kin(aid, px_sbar, py_sbar, pz_sbar, Ea, ma),
        _row_kin(did, px_us, py_us, pz_us, Ed, mdq),
    )


def _row_kin(pdg: int, px: float, py: float, pz: float, E: float, m: float) -> Dict[str, float]:
    pT, eta, phi, mcalc = kinematic_extras(px, py, pz, E)
    return {"px": px, "py": py, "pz": pz, "E": E, "m": mcalc, "pT": pT, "eta": eta, "phi": phi}


def replace_diquark_three_body(
    g: pd.DataFrame,
    dq_row: pd.Series,
    channel: str,
    tag_a: int,
    tag_b: int,
    x_struck: float,
    p_proton_breit_pz: float,
) -> Tuple[pd.DataFrame, int, int, int, Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Returns (df, idx_stripped, idx_antiq, idx_daughter, kin_s, kin_a, kin_d)."""
    dq_idx = int(dq_row["particle_index"])
    g2 = g[g["particle_index"] != dq_idx].copy()
    max_idx = int(g2["particle_index"].max()) if len(g2) else 0
    i1, i2, i3 = max_idx + 1, max_idx + 2, max_idx + 3
    kin_s, kin_a, kin_d = build_three_body_from_diquark(
        dq_row, channel, x_struck, p_proton_breit_pz
    )
    sid, did, aid = channel_ab(channel)

    m1 = int(dq_row["mother1"])
    m2 = int(dq_row["mother2"])
    st = int(dq_row["status"])

    common = {
        "event_id": int(dq_row["event_id"]),
        "status": st,
        "mother1": m1,
        "mother2": m2,
        "daughter1": 0,
        "daughter2": 0,
        "isFinal": 1,
    }
    stripped = {
        **common,
        "particle_index": i1,
        "pdg_id": sid,
        "col": tag_a,
        "acol": 0,
        **kin_s,
    }
    antiq = {
        **common,
        "particle_index": i2,
        "pdg_id": aid,
        "col": 0,
        "acol": tag_b,
        **kin_a,
    }
    daughter = {
        **common,
        "particle_index": i3,
        "pdg_id": did,
        "col": 0,
        "acol": tag_a,
        **kin_d,
    }
    out = pd.concat([g2, pd.DataFrame([stripped, antiq, daughter])], ignore_index=True)
    out = out.sort_values("particle_index").reset_index(drop=True)
    return out, i1, i2, i3, kin_s, kin_a, kin_d


def apply_px_kick_pair(df: pd.DataFrame, idx_a: int, idx_b: int, delta: float) -> pd.DataFrame:
    out = df.copy()
    for idx, sign in ((idx_a, 1.0), (idx_b, -1.0)):
        m = out["particle_index"] == idx
        if not m.any():
            raise RuntimeError(f"kick target index {idx} missing")
        row = out.loc[m].iloc[0]
        px = float(row["px"]) + sign * delta
        py, pz = float(row["py"]), float(row["pz"])
        m0 = float(row["m"])
        E = e_on_shell(px, py, pz, m0)
        pT, eta, phi, mcalc = kinematic_extras(px, py, pz, E)
        out.loc[m, ["px", "py", "pz", "E", "m", "pT", "eta", "phi"]] = [
            px,
            py,
            pz,
            E,
            mcalc,
            pT,
            eta,
            phi,
        ]
    return out


def write_event_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values("particle_index").to_csv(path, index=False)


def process_one_event_split(
    g: pd.DataFrame,
    event_id: int,
    *,
    event_in_alter_pool: bool,
    md_row: Optional[pd.Series],
    mode: str,
    delta_px: float,
    rng: random.Random,
    forced_split_channel: Optional[Literal["C"]] = None,
    seed_for_kick: int = 0,
) -> Tuple[pd.DataFrame, Literal["unchanged", "altered"], Optional[Dict[str, Any]], Dict[str, int]]:
    """
    Core 90/10 alteration physics (same as the CSV driver's per-event loop).

    Kick θ (v2): ``Random(f"{seed_for_kick}:{event_id}")`` — deterministic in serial and parallel.

    Returns:
        particles_out — sorted by ``particle_index``, ready to persist.
        branch — ``"unchanged"`` or ``"altered"``.
        altered_meta — surgery dict (old ``.meta.json`` payload) if ``branch=="altered"``, else ``None``.
        stats_delta — single non-zero counter increment for aggregation (e.g. ``written_unchanged``).
    """
    g = g.sort_values("particle_index").reset_index(drop=True)
    stats_delta: Dict[str, int] = {}

    struck_in = int(md_row["struck_incoming_index"]) if md_row is not None else -1

    if not event_in_alter_pool:
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    reco = find_outgoing_struck_quark_noisr(g, struck_in)
    if not bool(reco.get("success", False)) or reco.get("selected_row") is None:
        stats_delta["fallback_unchanged_struck_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    struck_row = reco["selected_row"]
    struck_idx = int(struck_row["particle_index"])

    if mode == "breit_px_kick_only":
        dq_row = find_final_diquark_kick_partner_row(g)
        if dq_row is None:
            stats_delta["fallback_unchanged_no_ud_diquark"] = 1
            stats_delta["written_unchanged"] = 1
            return g, "unchanged", None, stats_delta
        dq_idx = int(dq_row["particle_index"])
        if dq_idx == struck_idx:
            stats_delta["fallback_unchanged_validation_fail"] = 1
            stats_delta["written_unchanged"] = 1
            return g, "unchanged", None, stats_delta
        try:
            g_kick = apply_px_kick_pair(g.copy(), dq_idx, struck_idx, delta_px)
        except RuntimeError:
            stats_delta["fallback_unchanged_validation_fail"] = 1
            stats_delta["written_unchanged"] = 1
            return g, "unchanged", None, stats_delta
        meta = {
            "original_event_id": int(event_id),
            "mode": "breit_px_kick_only",
            "struck_quark_index": struck_idx,
            "kick_partner_particle_index": dq_idx,
            "kick_delta_px_gev": delta_px,
        }
        stats_delta["written_altered"] = 1
        return g_kick.sort_values("particle_index").reset_index(drop=True), "altered", meta, stats_delta

    dq_row = find_final_ud0_diquark(g)
    if dq_row is None:
        stats_delta["fallback_unchanged_no_ud_diquark"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    # v2: read Bjorken x of the struck quark and reject if too large
    if md_row is None or "x" not in md_row.index:
        stats_delta["fallback_unchanged_x_unavailable"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    x_struck = float(md_row["x"])
    if not math.isfinite(x_struck) or x_struck <= 0:
        stats_delta["fallback_unchanged_x_invalid"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    if x_struck > 0.495:
        stats_delta["fallback_unchanged_x_too_large"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    # v2: compute proton pz in the Breit frame from flipped-lab beam vectors
    from jet_hadron_observables_split_pi_pm import (  # noqa: WPS433
        flipped_lab_beam_arrays_from_event_row,
        kinematics_lab_to_breit_lt_from_flip_lab_beams,
    )
    import numpy as np

    blt = flipped_lab_beam_arrays_from_event_row(md_row)
    if blt is None:
        stats_delta["fallback_unchanged_no_beam_vectors"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    LT, _err = kinematics_lab_to_breit_lt_from_flip_lab_beams(
        *blt,
        x_exclusive_floor=0.0,
        q_open_interval_gev=(0.01, 1.0e6),
    )
    if LT is None:
        stats_delta["fallback_unchanged_LT_build_failed"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    _, _, pin_fl = blt
    p_proton_breit = LT @ pin_fl
    p_proton_breit_pz = float(p_proton_breit[3])

    channel = "C" if forced_split_channel is None else forced_split_channel
    if channel != "C":
        raise ValueError("only split channel C is supported")
    stats_delta["altered_channel_C"] = 1

    B = int(dq_row["acol"])
    if B <= 0:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    g_pre = g[g["particle_index"] != int(dq_row["particle_index"])].copy()
    tag_a = next_free_color_tag(g_pre)

    g_alt, i_strip, i_aq, i_dau, ks, ka, kd = replace_diquark_three_body(
        g, dq_row, channel, tag_a, B, x_struck, p_proton_breit_pz
    )

    px_dq, py_dq = float(dq_row["px"]), float(dq_row["py"])
    sx = ks["px"] + ka["px"] + kd["px"] - px_dq
    sy = ks["py"] + ka["py"] + kd["py"] - py_dq
    if max(abs(sx), abs(sy)) > TOL_3M:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    px_before = float(g_alt[g_alt["particle_index"] == i_strip]["px"].iloc[0]) + float(
        g_alt[g_alt["particle_index"] == struck_idx]["px"].iloc[0]
    )

    # Per-event seeded RNG: deterministic across serial and parallel runs.
    event_rng = random.Random(f"{int(seed_for_kick)}:{int(event_id)}")
    theta_rad = event_rng.uniform(0.0, math.pi)
    kick_magnitude = float(delta_px) * math.sin(theta_rad)
    try:
        g_kick = apply_px_kick_pair(g_alt, i_strip, struck_idx, kick_magnitude)
    except RuntimeError:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    px_after = float(g_kick[g_kick["particle_index"] == i_strip]["px"].iloc[0]) + float(
        g_kick[g_kick["particle_index"] == struck_idx]["px"].iloc[0]
    )
    if abs(px_after - px_before) > TOL_3M:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    ok_c, msg_c = validate_surgery_color_tags(g_kick, tag_a, B)
    if not ok_c:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    onshell_ok = True
    for idx in (i_strip, i_aq, i_dau, struck_idx):
        r = g_kick[g_kick["particle_index"] == idx].iloc[0]
        px, py, pz, E = float(r["px"]), float(r["py"]), float(r["pz"]), float(r["E"])
        m = float(r["m"])
        if abs(E * E - px * px - py * py - pz * pz - m * m) > TOL_ONSHELL:
            onshell_ok = False
            break
    if not onshell_ok:
        stats_delta["fallback_unchanged_validation_fail"] = 1
        stats_delta["written_unchanged"] = 1
        return g, "unchanged", None, stats_delta

    row_s_pre = g_alt[g_alt["particle_index"] == i_strip].iloc[0]
    row_st_pre = g_alt[g_alt["particle_index"] == struck_idx].iloc[0]
    stripped_px_before_kick = float(row_s_pre["px"])
    struck_px_before_kick = float(row_st_pre["px"])

    sid, did, aid = channel_ab(channel)
    meta = {
        "original_event_id": int(event_id),
        "split_channel": channel,
        "channel_description": "(ud)->[d]+(us)+sbar",
        "original_diquark_index": int(dq_row["particle_index"]),
        "original_diquark_pdg": int(dq_row["pdg_id"]),
        "original_diquark_px": float(dq_row["px"]),
        "original_diquark_py": float(dq_row["py"]),
        "original_diquark_pz": float(dq_row["pz"]),
        "struck_quark_index": struck_idx,
        "kick_delta_px_gev": delta_px,
        "kick_theta_rad": float(theta_rad),
        "kick_magnitude_gev": float(kick_magnitude),
        "x_struck_used": float(x_struck),
        "stripped_px_before_kick": stripped_px_before_kick,
        "struck_px_before_kick": struck_px_before_kick,
        "stripped_quark_index": i_strip,
        "created_antiquark_index": i_aq,
        "daughter_diquark_index": i_dau,
        "new_particle_pdgs": {"stripped": sid, "antiquark": aid, "daughter_diquark": did},
        "color_tag_A": tag_a,
        "color_tag_B": B,
    }
    stats_delta["written_altered"] = 1
    return g_kick.sort_values("particle_index").reset_index(drop=True), "altered", meta, stats_delta


def main() -> None:
    ap = argparse.ArgumentParser(description="90/10 DIS split: unchanged vs diquark-split+kick altered.")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for 90/10 assignment.")
    ap.add_argument("--delta-px", type=float, default=0.4, help="Balanced px kick (GeV) on stripped d and struck quark.")
    ap.add_argument("--out-root", type=Path, default=DEFAULT_SPLIT_ROOT, help="Output root (contains unchanged/ altered/).")
    ap.add_argument("--max-events", type=int, default=0, help="If >0, only process first N event_ids (for testing).")
    ap.add_argument(
        "--full-event-csv",
        type=Path,
        default=FULL_EVENT_CSV,
        help="Source full event record CSV.",
    )
    ap.add_argument(
        "--metadata-csv",
        type=Path,
        default=METADATA_CSV,
        help="Source event metadata CSV.",
    )
    ap.add_argument(
        "--mode",
        choices=("diquark_kick", "breit_px_kick_only"),
        default="diquark_kick",
        help=(
            "diquark_kick: replace (ud)0 diquark by 3-body then ±px kick (legacy). "
            "breit_px_kick_only: no topology change; ±px on struck quark + diquark partner "
            "in the CSV frame (use with Breit-frame generator output)."
        ),
    )
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_root: Path = args.out_root
    unchanged_dir = out_root / "unchanged"
    altered_dir = out_root / "altered"
    unchanged_dir.mkdir(parents=True, exist_ok=True)
    altered_dir.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(args.full_event_csv).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    md = pd.read_csv(args.metadata_csv).sort_values("event_id").reset_index(drop=True)
    md_map = md.set_index("event_id")

    all_ids = sorted(ev["event_id"].unique().tolist())
    if args.max_events > 0:
        all_ids = all_ids[: args.max_events]

    ids_shuffled = list(all_ids)
    rng.shuffle(ids_shuffled)
    n_alter = int(round(0.1 * len(ids_shuffled)))
    alter_set = set(ids_shuffled[:n_alter])
    unchanged_set = set(ids_shuffled[n_alter:])

    stats = {
        "mode": args.mode,
        "seed": args.seed,
        "delta_px": args.delta_px,
        "total_ids": len(all_ids),
        "planned_altered": n_alter,
        "planned_unchanged": len(unchanged_set),
        "written_unchanged": 0,
        "written_altered": 0,
        "altered_channel_C": 0,
        "fallback_unchanged_no_ud_diquark": 0,
        "fallback_unchanged_struck_fail": 0,
        "fallback_unchanged_validation_fail": 0,
    }

    for event_id in all_ids:
        g = ev[ev["event_id"] == event_id].copy()
        md_row = md_map.loc[event_id] if event_id in md_map.index else None
        particles_out, branch, altered_meta, delta = process_one_event_split(
            g,
            int(event_id),
            event_in_alter_pool=(event_id in alter_set),
            md_row=md_row,
            mode=str(args.mode),
            delta_px=float(args.delta_px),
            rng=rng,
        )
        for k, v in delta.items():
            stats[k] = stats.get(k, 0) + v
        if branch == "unchanged":
            write_event_csv(particles_out, unchanged_dir / f"event_{int(event_id):06d}.csv")
        else:
            alt_path = altered_dir / f"event_{int(event_id):06d}.csv"
            meta_path = altered_dir / f"event_{int(event_id):06d}.meta.json"
            write_event_csv(particles_out, alt_path)
            if altered_meta is not None:
                meta_path.write_text(json.dumps(altered_meta, indent=2), encoding="utf-8")

    stats["total_input_events"] = int(stats["total_ids"])
    stats["attempted_altered"] = int(stats["planned_altered"])
    stats["failed_altered"] = int(stats["attempted_altered"] - stats["written_altered"])

    summary_path = out_root / "split_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    n_files, capped = count_files_under(out_root)
    manifest_path = write_run_manifest(
        run_label="split_dis_sample_diquark_kick",
        script_name="scripts/analysis/split_dis_sample_diquark_kick.py",
        top_level_dirs_written=[str(out_root)],
        approximate_files_created=n_files,
        approximate_files_capped=capped,
        extra={"split_summary_json": str(summary_path)},
    )

    print("=== split_dis_sample_diquark_kick ===")
    print(f"out_root={out_root}")
    print(f"total event ids processed: {stats['total_ids']}")
    print(f"planned altered (10%): {stats['planned_altered']}; written altered: {stats['written_altered']}")
    print(f"planned unchanged (90%): {stats['planned_unchanged']}; written unchanged rows: {stats['written_unchanged']}")
    print(f"altered channel C (ud->d+su0+sbar): {stats.get('altered_channel_C', 0)}")
    print(
        f"fallback to unchanged (no ud diquark): {stats['fallback_unchanged_no_ud_diquark']}, "
        f"(struck fail): {stats['fallback_unchanged_struck_fail']}, "
        f"(validation): {stats['fallback_unchanged_validation_fail']}"
    )
    print(f"summary_json={summary_path}")
    print(f"run_manifest={manifest_path}")


if __name__ == "__main__":
    main()

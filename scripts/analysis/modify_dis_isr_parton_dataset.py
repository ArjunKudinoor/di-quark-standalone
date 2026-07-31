#!/usr/bin/env python3
"""
Stage 2 (PoP): replace final-state diquark remnant with explicit u+d pair.

This is a hand-built proof-of-principle remnant model:
  - We DO NOT extract remnant constituents from PYTHIA internals here.
  - We DO NOT apply any transverse kicks here.
  - We DO keep struck-quark momentum unchanged.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import outputs_dir

OUTDIR = outputs_dir() / "dis_isr_parton_dataset"
FULL_EVENT_CSV = OUTDIR / "dis_isr_full_event_record.csv"
METADATA_CSV = OUTDIR / "dis_isr_event_metadata.csv"
OUT_SPLIT_EVENT_CSV = OUTDIR / "dis_isr_full_event_record_diquark_split.csv"
OUT_SPLIT_META_CSV = OUTDIR / "dis_isr_diquark_split_metadata.csv"


def get_event_particles(df_event: pd.DataFrame) -> Dict[int, pd.Series]:
    return {int(r["particle_index"]): r for _, r in df_event.iterrows()}


def _direct_daughters_from_row(row: pd.Series) -> List[int]:
    d1 = int(row["daughter1"])
    d2 = int(row["daughter2"])
    if d1 > 0 and d2 >= d1:
        return list(range(d1, d2 + 1))
    return []


def build_descendant_set(df_event: pd.DataFrame, start_index: int) -> Set[int]:
    by_idx = get_event_particles(df_event)
    if start_index not in by_idx:
        return set()
    descendants: Set[int] = set()
    visited: Set[int] = set()
    queue: List[int] = [start_index]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        row = by_idx.get(cur)
        if row is None:
            continue
        for d in _direct_daughters_from_row(row):
            if d not in by_idx:
                continue
            if d not in descendants:
                descendants.add(d)
                queue.append(d)
    return descendants


def fourvec_from_row(row: pd.Series) -> Tuple[float, float, float, float]:
    return (float(row["E"]), float(row["px"]), float(row["py"]), float(row["pz"]))


def minkowski_square(E: float, px: float, py: float, pz: float) -> float:
    return E * E - px * px - py * py - pz * pz


def minkowski_diff_sq(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    return minkowski_square(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def vec_sub(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def vec_add(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def find_incoming_outgoing_electron(df_event: pd.DataFrame) -> Tuple[pd.Series | None, pd.Series | None]:
    in_cand = df_event[(df_event["pdg_id"] == 11) & (df_event["status"] < 0)]
    out_cand = df_event[(df_event["pdg_id"] == 11) & (df_event["isFinal"] == 1)]
    in_row = None if in_cand.empty else in_cand.loc[in_cand["pz"].idxmax()]
    out_row = None if out_cand.empty else out_cand.loc[out_cand["E"].idxmax()]
    return in_row, out_row


def find_outgoing_struck_quark_noisr(df_event: pd.DataFrame, struck_incoming_index: int) -> Dict[str, object]:
    by_idx = get_event_particles(df_event)
    struck_row = by_idx.get(struck_incoming_index)
    descendants = build_descendant_set(df_event, struck_incoming_index)
    desc_rows = [by_idx[i] for i in sorted(descendants) if i in by_idx]
    if struck_row is None:
        return {"success": False, "failure_reason": "struck_incoming_missing", "selected_row": None, "selection_mode": "", "descendants": descendants}
    candidates = [r for r in desc_rows if int(r["isFinal"]) == 1 and abs(int(r["pdg_id"])) in {1, 2, 3, 4, 5, 6}]
    if not candidates:
        return {"success": False, "failure_reason": "no_final_quark_descendants", "selected_row": None, "selection_mode": "", "descendants": descendants}
    if len(candidates) == 1:
        return {"success": True, "failure_reason": "", "selected_row": candidates[0], "selection_mode": "unique", "descendants": descendants}

    e_in, e_out = find_incoming_outgoing_electron(df_event)
    if e_in is None or e_out is None:
        return {"success": False, "failure_reason": "electron_line_missing", "selected_row": None, "selection_mode": "", "descendants": descendants}
    q = vec_sub(fourvec_from_row(e_in), fourvec_from_row(e_out))
    target = vec_add(fourvec_from_row(struck_row), q)
    metrics = [{"row": r, "absDelta2": abs(minkowski_diff_sq(fourvec_from_row(r), target))} for r in candidates]
    best = min(metrics, key=lambda x: float(x["absDelta2"]))
    return {"success": True, "failure_reason": "", "selected_row": best["row"], "selection_mode": "invariant_tiebreak", "descendants": descendants}


def find_final_state_diquark(df_event: pd.DataFrame) -> Tuple[pd.Series | None, int]:
    # Simple explicit rule for this PoP: final-state diquark-like IDs only.
    cand = df_event[(df_event["isFinal"] == 1) & (df_event["pdg_id"].abs() >= 1000) & (df_event["pdg_id"].abs() < 10000)]
    n = int(len(cand))
    if n == 0:
        return None, 0
    if n == 1:
        return cand.iloc[0], 1
    # Temporary rule: pick most energetic if multiple.
    chosen = cand.loc[cand["E"].idxmax()]
    return chosen, n


def _eta_from(px: float, py: float, pz: float) -> float:
    pt = math.hypot(px, py)
    if pt < 1e-12:
        if pz > 0:
            return 1e6
        if pz < 0:
            return -1e6
        return 0.0
    return math.asinh(pz / pt)


def split_diquark_equal_collinear(diquark_row: pd.Series) -> Tuple[Dict[str, float], Dict[str, float]]:
    # Hand-built model: equal 50/50 collinear split, no kicks.
    px = 0.5 * float(diquark_row["px"])
    py = 0.5 * float(diquark_row["py"])
    pz = 0.5 * float(diquark_row["pz"])
    E = 0.5 * float(diquark_row["E"])
    pT = math.hypot(px, py)
    eta = _eta_from(px, py, pz)
    phi = math.atan2(py, px)
    base = {"px": px, "py": py, "pz": pz, "E": E, "m": 0.0, "pT": pT, "eta": eta, "phi": phi}
    return base.copy(), base.copy()


def rewrite_event_with_split_diquark(df_event: pd.DataFrame, diquark_row: pd.Series, u_row: Dict[str, float], d_row: Dict[str, float]) -> Tuple[pd.DataFrame, int, int]:
    out = df_event.copy()
    dq_idx = int(diquark_row["particle_index"])
    out = out[out["particle_index"] != dq_idx].copy()
    max_idx = int(out["particle_index"].max()) if len(out) else -1
    new_u_idx = max_idx + 1
    new_d_idx = max_idx + 2

    # Placeholder ancestry/status/color scheme for CSV-level PoP only.
    m1 = int(diquark_row["mother1"])
    m2 = int(diquark_row["mother2"])
    status = int(diquark_row["status"])
    col = int(diquark_row["col"])
    acol = int(diquark_row["acol"])
    common = {
        "event_id": int(diquark_row["event_id"]),
        "status": status,
        "mother1": m1,
        "mother2": m2,
        "daughter1": 0,
        "daughter2": 0,
        "isFinal": 1,
    }
    u_new = {
        **common,
        "particle_index": new_u_idx,
        "pdg_id": 2,
        "col": col,
        "acol": 0,
        **u_row,
    }
    d_new = {
        **common,
        "particle_index": new_d_idx,
        "pdg_id": 1,
        "col": 0,
        "acol": acol,
        **d_row,
    }
    out = pd.concat([out, pd.DataFrame([u_new, d_new])], ignore_index=True)
    out = out.sort_values("particle_index").reset_index(drop=True)
    return out, new_u_idx, new_d_idx


def main() -> None:
    parser = argparse.ArgumentParser(description="Split no-ISR final-state diquark into explicit u+d pair.")
    parser.add_argument("--inspect-events", type=int, default=20, help="Print detailed diagnostics for first N events.")
    args = parser.parse_args()

    ev = pd.read_csv(FULL_EVENT_CSV).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    md = pd.read_csv(METADATA_CSV).sort_values("event_id").reset_index(drop=True)
    md_map = md.set_index("event_id")

    out_rows: List[pd.DataFrame] = []
    split_meta_rows: List[Dict[str, object]] = []

    total = 0
    n_struck_unique = 0
    n_dq_found = 0
    n_split_success = 0
    n_failed = 0
    n_multi_dq = 0
    diff_mags: List[float] = []

    for event_id, g in ev.groupby("event_id", sort=True):
        total += 1
        g = g.sort_values("particle_index").reset_index(drop=True)
        success = 0
        notes = ""
        struck_in = -1
        struck_out_idx = -1
        dq_idx = -1
        dq_id = 0
        u_idx = -1
        d_idx = -1

        if event_id in md_map.index:
            struck_in = int(md_map.loc[event_id]["struck_incoming_index"])

        reco = find_outgoing_struck_quark_noisr(g, struck_in)
        struck_out = reco.get("selected_row", None)
        if bool(reco.get("success", False)) and struck_out is not None:
            struck_out_idx = int(struck_out["particle_index"])
            n_struck_unique += 1
        else:
            notes = f"struck_fail:{reco.get('failure_reason','unknown')}"
            n_failed += 1
            out_rows.append(g)
            split_meta_rows.append(
                {
                    "event_id": int(event_id),
                    "struck_incoming_index": struck_in,
                    "struck_outgoing_index_selected": -1,
                    "original_diquark_index": -1,
                    "original_diquark_pdg_id": 0,
                    "original_diquark_px": "",
                    "original_diquark_py": "",
                    "original_diquark_pz": "",
                    "original_diquark_E": "",
                    "new_u_index": -1,
                    "new_d_index": -1,
                    "new_u_px": "",
                    "new_u_py": "",
                    "new_u_pz": "",
                    "new_u_E": "",
                    "new_d_px": "",
                    "new_d_py": "",
                    "new_d_pz": "",
                    "new_d_E": "",
                    "split_mode": "equal_collinear_ud",
                    "success": 0,
                    "notes": notes,
                }
            )
            continue

        dq_row, n_dq = find_final_state_diquark(g)
        if dq_row is None:
            notes = "no_final_state_diquark"
            n_failed += 1
            out_rows.append(g)
            split_meta_rows.append(
                {
                    "event_id": int(event_id),
                    "struck_incoming_index": struck_in,
                    "struck_outgoing_index_selected": struck_out_idx,
                    "original_diquark_index": -1,
                    "original_diquark_pdg_id": 0,
                    "original_diquark_px": "",
                    "original_diquark_py": "",
                    "original_diquark_pz": "",
                    "original_diquark_E": "",
                    "new_u_index": -1,
                    "new_d_index": -1,
                    "new_u_px": "",
                    "new_u_py": "",
                    "new_u_pz": "",
                    "new_u_E": "",
                    "new_d_px": "",
                    "new_d_py": "",
                    "new_d_pz": "",
                    "new_d_E": "",
                    "split_mode": "equal_collinear_ud",
                    "success": 0,
                    "notes": notes,
                }
            )
            continue

        n_dq_found += 1
        if n_dq > 1:
            n_multi_dq += 1
        dq_idx = int(dq_row["particle_index"])
        dq_id = int(dq_row["pdg_id"])
        u_new, d_new = split_diquark_equal_collinear(dq_row)
        g_new, u_idx, d_idx = rewrite_event_with_split_diquark(g, dq_row, u_new, d_new)
        out_rows.append(g_new)
        success = 1
        n_split_success += 1

        dpx = (u_new["px"] + d_new["px"]) - float(dq_row["px"])
        dpy = (u_new["py"] + d_new["py"]) - float(dq_row["py"])
        dpz = (u_new["pz"] + d_new["pz"]) - float(dq_row["pz"])
        dE = (u_new["E"] + d_new["E"]) - float(dq_row["E"])
        diff = math.sqrt(dpx * dpx + dpy * dpy + dpz * dpz + dE * dE)
        diff_mags.append(diff)

        split_meta_rows.append(
            {
                "event_id": int(event_id),
                "struck_incoming_index": struck_in,
                "struck_outgoing_index_selected": struck_out_idx,
                "original_diquark_index": dq_idx,
                "original_diquark_pdg_id": dq_id,
                "original_diquark_px": float(dq_row["px"]),
                "original_diquark_py": float(dq_row["py"]),
                "original_diquark_pz": float(dq_row["pz"]),
                "original_diquark_E": float(dq_row["E"]),
                "new_u_index": u_idx,
                "new_d_index": d_idx,
                "new_u_px": u_new["px"],
                "new_u_py": u_new["py"],
                "new_u_pz": u_new["pz"],
                "new_u_E": u_new["E"],
                "new_d_px": d_new["px"],
                "new_d_py": d_new["py"],
                "new_d_pz": d_new["pz"],
                "new_d_E": d_new["E"],
                "split_mode": "equal_collinear_ud",
                "success": success,
                "notes": "hand_built_equal_split_no_kick",
            }
        )

        if total <= args.inspect_events:
            print(f"--- event_id={event_id} ---")
            print(f"struck_incoming_index={struck_in}")
            print(
                "selected outgoing struck quark: "
                f"idx={struck_out_idx}, pdg_id={int(struck_out['pdg_id'])}, "
                f"p=({float(struck_out['px'])}, {float(struck_out['py'])}, {float(struck_out['pz'])}, {float(struck_out['E'])})"
            )
            print(
                "selected original diquark: "
                f"idx={dq_idx}, pdg_id={dq_id}, "
                f"p=({float(dq_row['px'])}, {float(dq_row['py'])}, {float(dq_row['pz'])}, {float(dq_row['E'])})"
            )
            print(f"constructed u quark: idx={u_idx}, pdg_id=2, p=({u_new['px']}, {u_new['py']}, {u_new['pz']}, {u_new['E']})")
            print(f"constructed d quark: idx={d_idx}, pdg_id=1, p=({d_new['px']}, {d_new['py']}, {d_new['pz']}, {d_new['E']})")
            print(f"check p_u+p_d-p_dq: ({dpx}, {dpy}, {dpz}, {dE})")
            print(f"success={success}")

    out_df = pd.concat(out_rows, ignore_index=True).sort_values(["event_id", "particle_index"]).reset_index(drop=True)
    split_meta_df = pd.DataFrame(split_meta_rows).sort_values("event_id").reset_index(drop=True)
    out_df.to_csv(OUT_SPLIT_EVENT_CSV, index=False)
    split_meta_df.to_csv(OUT_SPLIT_META_CSV, index=False)

    avg_diff = sum(diff_mags) / len(diff_mags) if diff_mags else float("nan")
    max_diff = max(diff_mags) if diff_mags else float("nan")

    print("\n=== Summary ===")
    print(f"total events processed={total}")
    print(f"number with a uniquely identified struck quark={n_struck_unique}")
    print(f"number with a diquark found={n_dq_found}")
    print(f"number successfully split={n_split_success}")
    print(f"number failed={n_failed}")
    print(f"number of events with multiple diquark candidates={n_multi_dq}")
    print(f"average |p_u+p_d-p_dq|_4diff={avg_diff}")
    print(f"max |p_u+p_d-p_dq|_4diff={max_diff}")
    print(f"output_event_record={OUT_SPLIT_EVENT_CSV}")
    print(f"output_metadata={OUT_SPLIT_META_CSV}")


if __name__ == "__main__":
    main()


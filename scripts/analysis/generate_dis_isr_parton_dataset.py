#!/usr/bin/env python3
"""
Stage 1 (restructured): build a full DIS event-record dataset for struck-u events.

For each accepted event:
  - Save the full PYTHIA event record (all particles, all statuses)
  - Save event-level metadata (Q2, xB, struck indices)

By default, px, py, pz, E, m, pT, eta, phi are written in the same Breit-like frame as
``diquark.analyze_events_raw.build_LT`` (after the beam ``flip_z`` convention used for
pTrel analysis): incoming proton and virtual-photon exchange axis lie along ±z with
opposite signs. Use ``--no-breit-frame`` for lab-frame momenta (legacy behavior).

Downstream code that applies ``build_LT`` again to these CSV rows would double-transform;
either use lab output or treat CSV momenta as already Breit.

Optional: with ``--kick-fraction`` > 0 (default 0.1), a Bernoulli subset of events receive a
balanced transverse kick on the outgoing struck quark (same identification as
``modify_dis_isr_parton_dataset.find_outgoing_struck_quark_noisr``) and a diquark partner:
prefer final ud0 (PDG 2101); if missing, any final diquark-like line (1000 <= |pdg| < 10000),
highest E. One leg gets +(kx, ky), the other −(kx, ky), with |k| = ``--kick-kt-gev`` and random
azimuth; energies are recomputed on-shell at fixed rest mass. If no partner is found, the event
is left unkicked (counts toward ``kick_rolls_missing_pair``).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pythia8

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from diquark.paths import count_files_under, outputs_dir, write_run_manifest

from diquark.analyze_events_raw import FLIP_Z_PTREL, build_LT, flip_z  # noqa: E402

OUTDIR = outputs_dir() / "dis_isr_parton_dataset"
DEFAULT_N_ACCEPTED = 100_000
WRITE_CHUNK_ROWS = 50_000
DIQUARK_UD0 = 2101

# One event row written to CSV: indices for kick helpers
_RI_PID = 2
_RI_PX = 10
_RI_PY = 11
_RI_PZ = 12
_RI_E = 13
_RI_M = 14
_RI_PT = 15
_RI_ETA = 16
_RI_PHI = 17
_RI_FINAL = 18


def build_pythia() -> pythia8.Pythia:
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString("Beams:eA = 18.0")
    p.readString("Beams:eB = 275.0")
    p.readString("Beams:frameType = 2")

    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString("PhaseSpace:Q2Min = 16.0")

    p.readString("ProcessLevel:all = on")
    p.readString("PartonLevel:all = on")
    p.readString("PDF:lepton = off")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = off")
    p.readString("PartonLevel:MPI = off")
    p.readString("PartonLevel:Remnants = on")
    p.readString("HadronLevel:all = off")

    p.readString("Random:setSeed = on")
    p.readString("Random:seed = 12345")

    p.readString("Print:quiet = on")

    p.init()
    return p


def pick_incoming_quark_index(ev: pythia8.Event) -> int | None:
    """
    Simplified selection requested:
      - incoming (status < 0)
      - quark (abs(id) <= 6)
      - pick one with largest |pz|
    """
    best_idx = None
    best_abs_pz = -1.0
    for i in range(ev.size()):
        p = ev[i]
        if p.status() >= 0:
            continue
        if abs(p.id()) > 6:
            continue
        apz = abs(float(p.pz()))
        if apz > best_abs_pz:
            best_abs_pz = apz
            best_idx = i
    return best_idx


def is_finite_particle(px: float, py: float, pz: float, E: float, m: float) -> bool:
    return all(math.isfinite(v) for v in (px, py, pz, E, m))


def _p4_from_particle(p: Any) -> np.ndarray:
    return np.array(
        [float(p.e()), float(p.px()), float(p.py()), float(p.pz())],
        dtype=np.float64,
    )


def extract_beams_from_event(
    ev: pythia8.Event,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Lab-frame beam four-vectors [E, px, py, pz], same selection as pTrel shard analysis:
    incoming e- (status < 0, max pz), scattered e- (final, max E), incoming proton (2212,
    status < 0, max pz).
    """
    in_e: List[Tuple[float, np.ndarray]] = []
    out_e: List[Tuple[float, np.ndarray]] = []
    in_p: List[Tuple[float, np.ndarray]] = []
    for i in range(ev.size()):
        p = ev[i]
        pid = int(p.id())
        st = int(p.status())
        v = _p4_from_particle(p)
        if pid == 11 and st < 0:
            in_e.append((float(p.pz()), v))
        if pid == 11 and p.isFinal():
            out_e.append((float(p.e()), v))
        if pid == 2212 and st < 0:
            in_p.append((float(p.pz()), v))
    if not in_e or not out_e or not in_p:
        return None
    e_in = max(in_e, key=lambda t: t[0])[1]
    e_sc = max(out_e, key=lambda t: t[0])[1]
    p_in = max(in_p, key=lambda t: t[0])[1]
    return e_in, e_sc, p_in


def try_build_lt_from_event(ev: pythia8.Event) -> Optional[np.ndarray]:
    """Return 4×4 Lorentz matrix lab→Breit (after flip_z), or None if kinematics fail."""
    beams = extract_beams_from_event(ev)
    if beams is None:
        return None
    e_in, e_sc, p_in = beams
    e_in_ev = flip_z(e_in, FLIP_Z_PTREL)
    e_sc_ev = flip_z(e_sc, FLIP_Z_PTREL)
    p_in_ev = flip_z(p_in, FLIP_Z_PTREL)
    qmu = e_in_ev - e_sc_ev
    q0, q1, q2, q3 = float(qmu[0]), float(qmu[1]), float(qmu[2]), float(qmu[3])
    Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
    if Q2 <= 0.0:
        return None
    qT = math.hypot(q1, q2)
    p_dot_q = (
        float(p_in_ev[0]) * q0
        - float(p_in_ev[1]) * q1
        - float(p_in_ev[2]) * q2
        - float(p_in_ev[3]) * q3
    )
    if p_dot_q == 0.0:
        return None
    x = Q2 / (2.0 * p_dot_q)
    Ee = float(e_in_ev[0])
    Ep = float(p_in_ev[0])
    S = 4.0 * Ee * Ep
    if S <= 0.0:
        return None
    y = Q2 / (S * x)
    phiq = math.atan2(q2, q1)
    return build_LT(Ee, Ep, np.array([q0, q1, q2, q3]), x, y, qT, phiq, S)


def kinematics_from_p4(E: float, px: float, py: float, pz: float) -> Tuple[float, float, float, float]:
    """Return (m, pT, eta, phi) from four-momentum; eta is pseudorapidity of the 3-momentum."""
    p2 = px * px + py * py + pz * pz
    m = math.sqrt(max(0.0, E * E - p2))
    pT = math.hypot(px, py)
    phi = math.atan2(py, px)
    p_mag = math.sqrt(max(0.0, p2))
    if p_mag <= 0.0:
        eta = 0.0
    else:
        den = p_mag - pz
        num = p_mag + pz
        if den <= 1e-18:
            eta = math.copysign(25.0, pz) if abs(pz) > 1e-18 else 0.0
        elif num <= 0.0:
            eta = 0.0
        else:
            eta = 0.5 * math.log(num / den)
    return m, pT, eta, phi


def e_on_shell(px: float, py: float, pz: float, m: float) -> float:
    return math.sqrt(max(0.0, px * px + py * py + pz * pz + m * m))


def kinematic_extras_split_style(px: float, py: float, pz: float, E: float) -> Tuple[float, float, float, float]:
    """Match split_dis_sample_diquark_kick.kinematic_extras (pT, eta, phi, m)."""
    pT = math.hypot(px, py)
    if pT < 1e-14:
        eta = math.copysign(1e6, pz) if pz != 0 else 0.0
    else:
        eta = math.asinh(pz / pT)
    phi = math.atan2(py, px)
    m = math.sqrt(max(0.0, E * E - px * px - py * py - pz * pz))
    return pT, eta, phi, m


def find_event_row_list_index_for_ud0_diquark(rows: List[Tuple[Any, ...]]) -> int | None:
    """List index of chosen final (ud0) diquark row; same heuristics as split script."""
    cand: List[Tuple[int, Tuple[Any, ...]]] = []
    for li, r in enumerate(rows):
        if int(r[_RI_PID]) != DIQUARK_UD0 or int(r[_RI_FINAL]) != 1:
            continue
        cand.append((li, r))
    if not cand:
        return None
    if len(cand) == 1:
        return cand[0][0]
    for li, r in cand:
        if int(r[1]) == 32:
            return li
    best_li, _ = max(cand, key=lambda t: float(t[1][_RI_E]))
    return best_li


def find_event_row_list_index_for_diquark_kick_partner(rows: List[Tuple[Any, ...]]) -> int | None:
    """
    Partner leg for balanced kick: prefer final ud0 (2101) like split_dis_sample_diquark_kick;
    if absent, any final diquark-like entry (1000 <= |pdg| < 10000), highest E.
    """
    li_ud0 = find_event_row_list_index_for_ud0_diquark(rows)
    if li_ud0 is not None:
        return li_ud0
    cand: List[Tuple[int, float]] = []
    for li, r in enumerate(rows):
        if int(r[_RI_FINAL]) != 1:
            continue
        ap = abs(int(r[_RI_PID]))
        if 1000 <= ap < 10000:
            cand.append((li, float(r[_RI_E])))
    if not cand:
        return None
    return max(cand, key=lambda t: t[1])[0]


def find_event_row_list_index_for_particle_index(rows: List[Tuple[Any, ...]], particle_index: int) -> int | None:
    for li, r in enumerate(rows):
        if int(r[1]) == particle_index:
            return li
    return None


def apply_balanced_transverse_kick_two_rows(
    rows: List[Tuple[Any, ...]],
    list_idx_a: int,
    list_idx_b: int,
    kx: float,
    ky: float,
) -> None:
    """In-place: row A += (kx,ky), row B -= (kx,ky); E and derived kin updated on-shell."""

    def bumped(r: Tuple[Any, ...], dpx: float, dpy: float) -> Tuple[Any, ...]:
        (
            eid,
            pi,
            pid,
            st,
            m1,
            m2,
            d1,
            d2,
            col,
            acol,
            px,
            py,
            pz,
            E,
            m,
            _pt,
            _eta,
            _phi,
            fin,
        ) = r
        px, py, pz, m = float(px), float(py), float(pz), float(m)
        npx, npy = px + dpx, py + dpy
        nE = e_on_shell(npx, npy, pz, m)
        npt, neta, nphi, nm = kinematic_extras_split_style(npx, npy, pz, nE)
        return (
            eid,
            pi,
            pid,
            st,
            m1,
            m2,
            d1,
            d2,
            col,
            acol,
            npx,
            npy,
            pz,
            nE,
            nm,
            npt,
            neta,
            nphi,
            fin,
        )

    rows[list_idx_a] = bumped(rows[list_idx_a], kx, ky)
    rows[list_idx_b] = bumped(rows[list_idx_b], -kx, -ky)


def _pythia_direct_daughters(p: Any) -> List[int]:
    d1 = int(p.daughter1())
    d2 = int(p.daughter2())
    if d1 > 0 and d2 >= d1:
        return list(range(d1, d2 + 1))
    return []


def build_descendant_indices_pythia(ev: pythia8.Event, start_idx: int) -> set[int]:
    """All particle indices reachable forward from start_idx via daughter links."""
    n = ev.size()
    if start_idx < 0 or start_idx >= n:
        return set()
    descendants: set[int] = set()
    visited: set[int] = set()
    queue: List[int] = [start_idx]
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        if cur < 0 or cur >= n:
            continue
        p = ev[cur]
        for d in _pythia_direct_daughters(p):
            if d < 0 or d >= n:
                continue
            if d not in descendants:
                descendants.add(d)
                queue.append(d)
    return descendants


def _pythia_fourvec_lab(p: Any) -> Tuple[float, float, float, float]:
    return (float(p.e()), float(p.px()), float(p.py()), float(p.pz()))


def _minkowski_square(E: float, px: float, py: float, pz: float) -> float:
    return E * E - px * px - py * py - pz * pz


def _minkowski_diff_sq(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> float:
    return _minkowski_square(a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def _vec_sub(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3])


def _vec_add(
    a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]
) -> Tuple[float, float, float, float]:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])


def find_incoming_outgoing_electron_pythia(ev: pythia8.Event) -> Tuple[int | None, int | None]:
    n = ev.size()
    best_in: Tuple[float, int] | None = None
    best_out: Tuple[float, int] | None = None
    for i in range(n):
        p = ev[i]
        if int(p.id()) != 11:
            continue
        if int(p.status()) < 0:
            pz = float(p.pz())
            if best_in is None or pz > best_in[0]:
                best_in = (pz, i)
        elif p.isFinal():
            E = float(p.e())
            if best_out is None or E > best_out[0]:
                best_out = (E, i)
    return (None if best_in is None else best_in[1], None if best_out is None else best_out[1])


def identify_struck_outgoing_quark_index(ev: pythia8.Event, struck_in_idx: int) -> int:
    """
    Outgoing struck quark: same logic as modify_dis_isr_parton_dataset.find_outgoing_struck_quark_noisr
    (descendants of incoming struck parton; lab-frame Minkowski tie-break vs k_in + q).
    Returns -1 if not found.
    """
    n = ev.size()
    if struck_in_idx < 0 or struck_in_idx >= n:
        return -1
    struck_p = ev[struck_in_idx]
    descendants = build_descendant_indices_pythia(ev, struck_in_idx)
    candidates: List[int] = []
    for i in sorted(descendants):
        if i < 0 or i >= n:
            continue
        p = ev[i]
        if not p.isFinal():
            continue
        if abs(int(p.id())) not in {1, 2, 3, 4, 5, 6}:
            continue
        candidates.append(i)
    if not candidates:
        return -1
    if len(candidates) == 1:
        return candidates[0]
    i_in, i_out = find_incoming_outgoing_electron_pythia(ev)
    if i_in is None or i_out is None:
        return -1
    qmu = _vec_sub(_pythia_fourvec_lab(ev[i_in]), _pythia_fourvec_lab(ev[i_out]))
    target = _vec_add(_pythia_fourvec_lab(struck_p), qmu)
    best_idx = candidates[0]
    best_d2 = float("inf")
    for i in candidates:
        d2 = abs(_minkowski_diff_sq(_pythia_fourvec_lab(ev[i]), target))
        if d2 < best_d2:
            best_d2 = d2
            best_idx = i
    return best_idx


# Hard subprocess in ``pythia.process`` (DIS NC): incoming documentation partons often have
# status -21; outgoing documentation products often have status 23 (see
# ``scripts/generation/convert_single_hard_process_to_lhe.py``). Read ``process`` right after
# the call that produced this subprocess — it is not guaranteed stable across later steps.


def hard_subprocess_incoming_quark_process_index(proc: pythia8.Event) -> int:
    """Process index of incoming hard quark (status -21, 1 <= |id| <= 6), or -1."""
    n = proc.size()
    for i in range(n):
        p = proc[i]
        if int(p.status()) != -21:
            continue
        pid = int(p.id())
        if 1 <= abs(pid) <= 6:
            return int(i)
    return -1


def hard_subprocess_outgoing_quark_lab_p4_and_index(
    proc: pythia8.Event,
) -> Tuple[Optional[np.ndarray], int]:
    """
    Outgoing quark from the hard subprocess (e.g. γ* + q → q + e): status 23 and quark PDG.

    Returns (four_vector_lab, process_index) with four-vector shape [E, px, py, pz], or
    (None, -1) if not found. If several status-23 quarks exist, keeps the highest-energy one.
    """
    n = proc.size()
    found: List[Tuple[int, Any]] = []
    for i in range(n):
        p = proc[i]
        if int(p.status()) != 23:
            continue
        pid = int(p.id())
        if 1 <= abs(pid) <= 6:
            found.append((int(i), p))
    if not found:
        return None, -1
    i_best, p_best = max(found, key=lambda t: float(t[1].e()))
    v = np.array(
        [float(p_best.e()), float(p_best.px()), float(p_best.py()), float(p_best.pz())],
        dtype=np.float64,
    )
    return v, int(i_best)


def _print_filtered_dir(obj: Any, title: str) -> None:
    keys = ["beam", "parton", "remnant", "resolve", "valence", "companion", "list", "id", "x", "p", "mother", "daughter"]
    names = [n for n in dir(obj) if any(k in n.lower() for k in keys)]
    print(f"[introspect] {title} type={type(obj)}")
    print(f"[introspect] {title} attrs={sorted(names)}")


def _safe_call(obj: Any, method_name: str) -> None:
    if not hasattr(obj, method_name):
        print(f"[introspect] {method_name}: missing")
        return
    fn = getattr(obj, method_name)
    if not callable(fn):
        print(f"[introspect] {method_name}: exists but not callable")
        return
    try:
        out = fn()
        print(f"[introspect] {method_name}() -> {out}")
    except Exception as exc:
        print(f"[introspect] {method_name}() failed: {exc}")


def _beam_introspection_once(pythia: pythia8.Pythia) -> None:
    print("=== Beam/remnant introspection (once) ===")
    _print_filtered_dir(pythia, "pythia")
    for name in ["beamA", "beamB", "beamAPtr", "beamBPtr", "process", "event", "infoPython"]:
        _safe_call(pythia, name)
    beam_b = None
    try:
        beam_b = pythia.beamB()
    except Exception as exc:
        print(f"[introspect] beamB() access failed: {exc}")
    if beam_b is None:
        print("[introspect] beamB object unavailable in Python binding")
        return
    _print_filtered_dir(beam_b, "beamB")
    for m in [
        "id", "size", "list", "xfISR", "nValence", "nCompanion", "isLepton",
        "isUnresolved", "isResolved", "xValence", "idValence", "idCompanion",
        "xCompanion", "idRemnant", "xRemnant", "p", "px", "py", "pz", "e",
    ]:
        _safe_call(beam_b, m)
    print("=== End introspection ===")


def _print_event_remnant_diagnostic(pythia: pythia8.Pythia, ev: pythia8.Event, event_id: int, inc_idx: int) -> None:
    inc = ev[inc_idx]
    print(f"\n--- accepted_event={event_id} ---")
    print(
        "struck incoming: "
        f"idx={inc_idx} id={inc.id()} status={inc.status()} "
        f"p=(px,py,pz,E)=({inc.px():.6f}, {inc.py():.6f}, {inc.pz():.6f}, {inc.e():.6f})"
    )
    proton_idx = -1
    for i in range(ev.size()):
        if int(ev[i].id()) == 2212 and int(ev[i].status()) < 0:
            proton_idx = i
            break
    if proton_idx >= 0:
        p = ev[proton_idx]
        print(
            "proton beam particle: "
            f"idx={proton_idx} id={p.id()} status={p.status()} "
            f"p=(px,py,pz,E)=({p.px():.6f}, {p.py():.6f}, {p.pz():.6f}, {p.e():.6f})"
        )
    else:
        print("proton beam particle: not found in event record")

    # Print remnant-like entries from event record as a proxy.
    remnant_like = []
    for i in range(ev.size()):
        pid = int(ev[i].id())
        if abs(pid) >= 1000 or int(ev[i].status()) in {63, -63}:
            remnant_like.append(i)
    print(f"event remnant-like entries indices={remnant_like}")
    for i in remnant_like:
        p = ev[i]
        print(
            f"  idx={i} id={p.id()} status={p.status()} mothers=({p.mother1()},{p.mother2()}) "
            f"daughters=({p.daughter1()},{p.daughter2()}) col={p.col()} acol={p.acol()} "
            f"p=(px,py,pz,E)=({p.px():.6f}, {p.py():.6f}, {p.pz():.6f}, {p.e():.6f})"
        )

    try:
        beam_b = pythia.beamB()
        print(f"beamB object type={type(beam_b)}")
        for m in ["size", "list", "nValence", "nCompanion", "idValence", "xValence", "idCompanion", "xCompanion"]:
            _safe_call(beam_b, m)
    except Exception as exc:
        print(f"beamB per-event access failed: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DIS full event record dataset.")
    parser.add_argument(
        "--inspect-remnant-internals",
        type=int,
        default=0,
        help="If >0, run tiny accepted-event diagnostic and print beam/remnant internals.",
    )
    parser.add_argument(
        "--n-accepted",
        type=int,
        default=DEFAULT_N_ACCEPTED,
        help=f"Target accepted struck-u events (default {DEFAULT_N_ACCEPTED}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTDIR,
        help=f"Directory for CSV outputs (default: {OUTDIR}).",
    )
    parser.add_argument(
        "--no-breit-frame",
        action="store_true",
        help="Write lab-frame momenta (no build_LT); default is Breit-like frame from build_LT.",
    )
    parser.add_argument(
        "--kick-fraction",
        type=float,
        default=0.1,
        help="Fraction of accepted events (Bernoulli) to apply balanced transverse kick (default 0.1). Use 0 to disable.",
    )
    parser.add_argument(
        "--kick-kt-gev",
        type=float,
        default=0.4,
        help="Magnitude |k| of kick vector on one leg; partner gets −k (default 0.4 GeV).",
    )
    parser.add_argument(
        "--kick-seed",
        type=int,
        default=98765,
        help="RNG seed for kick subsampling and azimuth (independent of PYTHIA seed).",
    )
    args = parser.parse_args()

    out_root = args.output_dir.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    out_event_record = out_root / "dis_isr_full_event_record.csv"
    out_metadata = out_root / "dis_isr_event_metadata.csv"
    n_target = int(args.n_accepted)

    # reset output files for a clean run
    if args.inspect_remnant_internals <= 0:
        if out_event_record.exists():
            out_event_record.unlink()
        if out_metadata.exists():
            out_metadata.unlink()

    pythia = build_pythia()
    ev = pythia.event

    total_generated = 0
    accepted_events = 0
    breit_rejections = 0
    use_breit = not bool(args.no_breit_frame)
    kick_fraction = min(1.0, max(0.0, float(args.kick_fraction)))
    kick_kt = max(0.0, float(args.kick_kt_gev))
    kick_rng = random.Random(int(args.kick_seed)) if kick_fraction > 0 and kick_kt > 0 else None
    kicks_applied = 0
    kicks_roll_no_pair = 0
    rows_buffer: List[
        Tuple[
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            int,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            float,
            int,
        ]
    ] = []
    metadata_rows: List[Tuple[int, float, float, int, int, int, float, float]] = []

    if args.inspect_remnant_internals > 0:
        _beam_introspection_once(pythia)
        target = int(args.inspect_remnant_internals)
        while accepted_events < target:
            if not pythia.next():
                continue
            total_generated += 1
            inc_idx = pick_incoming_quark_index(ev)
            if inc_idx is None:
                continue
            if abs(int(ev[inc_idx].id())) != 2:
                continue
            accepted_events += 1
            _print_event_remnant_diagnostic(pythia, ev, accepted_events, inc_idx)

        acceptance_fraction = accepted_events / total_generated if total_generated > 0 else float("nan")
        print("\n=== diagnostic summary ===")
        print(f"total_generated={total_generated}")
        print(f"total_accepted={accepted_events}")
        print(f"acceptance_fraction={acceptance_fraction:.6f}")
        print("Diagnostic mode complete; no CSV outputs written.")
        return

    with out_event_record.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "particle_index",
                "pdg_id",
                "status",
                "mother1",
                "mother2",
                "daughter1",
                "daughter2",
                "col",
                "acol",
                "px",
                "py",
                "pz",
                "E",
                "m",
                "pT",
                "eta",
                "phi",
                "isFinal",
            ]
        )

        while accepted_events < n_target:
            if not pythia.next():
                continue
            total_generated += 1

            inc_idx = pick_incoming_quark_index(ev)
            if inc_idx is None:
                continue

            inc_id = int(ev[inc_idx].id())
            if abs(inc_id) != 2:
                continue

            LT: Optional[np.ndarray] = None
            if use_breit:
                LT = try_build_lt_from_event(ev)
                if LT is None:
                    breit_rejections += 1
                    continue

            accepted_events += 1
            event_id = accepted_events
            struck_out_idx = identify_struck_outgoing_quark_index(ev, inc_idx)

            event_rows: List[
                Tuple[
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    int,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    float,
                    int,
                ]
            ] = []

            for i in range(ev.size()):
                p = ev[i]
                pid = int(p.id())
                status = int(p.status())
                m1 = int(p.mother1())
                m2 = int(p.mother2())
                d1 = int(p.daughter1())
                d2 = int(p.daughter2())
                col = int(p.col())
                acol = int(p.acol())
                px = float(p.px())
                py = float(p.py())
                pz = float(p.pz())
                E = float(p.e())
                is_final = int(bool(p.isFinal()))

                if E <= 0:
                    continue
                m = float(p.m())
                if not is_finite_particle(px, py, pz, E, m):
                    continue

                if LT is not None:
                    p4_lab = np.array([E, px, py, pz], dtype=np.float64)
                    p4_lab = flip_z(p4_lab, FLIP_Z_PTREL)
                    p4_b = LT @ p4_lab
                    E = float(p4_b[0])
                    px = float(p4_b[1])
                    py = float(p4_b[2])
                    pz = float(p4_b[3])
                    m, pt, eta, phi = kinematics_from_p4(E, px, py, pz)
                else:
                    m = float(p.m())
                    pt = float(p.pT())
                    eta = float(p.eta())
                    phi = float(p.phi())

                if E <= 0:
                    continue
                if not is_finite_particle(px, py, pz, E, m):
                    continue

                event_rows.append(
                    (
                        event_id,
                        i,
                        pid,
                        status,
                        m1,
                        m2,
                        d1,
                        d2,
                        col,
                        acol,
                        px,
                        py,
                        pz,
                        E,
                        m,
                        pt,
                        eta,
                        phi,
                        is_final,
                    )
                )

            kick_applied_flag = 0
            kick_kx_out = 0.0
            kick_ky_out = 0.0
            if kick_rng is not None and kick_rng.random() < kick_fraction:
                li_dq = find_event_row_list_index_for_diquark_kick_partner(event_rows)
                li_sq = (
                    find_event_row_list_index_for_particle_index(event_rows, struck_out_idx)
                    if struck_out_idx >= 0
                    else None
                )
                if li_dq is not None and li_sq is not None and li_dq != li_sq:
                    phi_k = kick_rng.uniform(0.0, 2.0 * math.pi)
                    kx = kick_kt * math.cos(phi_k)
                    ky = kick_kt * math.sin(phi_k)
                    apply_balanced_transverse_kick_two_rows(event_rows, li_sq, li_dq, kx, ky)
                    kick_applied_flag = 1
                    kick_kx_out = kx
                    kick_ky_out = ky
                    kicks_applied += 1
                else:
                    kicks_roll_no_pair += 1

            rows_buffer.extend(event_rows)

            info = pythia.infoPython()
            q2 = float(info.Q2Fac())
            xb = float(info.x2())
            metadata_rows.append(
                (
                    event_id,
                    q2,
                    xb,
                    int(inc_idx),
                    int(struck_out_idx),
                    kick_applied_flag,
                    kick_kx_out,
                    kick_ky_out,
                )
            )

            if len(rows_buffer) >= WRITE_CHUNK_ROWS:
                writer.writerows(rows_buffer)
                rows_buffer.clear()

        if rows_buffer:
            writer.writerows(rows_buffer)

    with out_metadata.open("w", newline="") as mf:
        mwriter = csv.writer(mf)
        mwriter.writerow(
            [
                "event_id",
                "Q2",
                "xB",
                "struck_incoming_index",
                "struck_outgoing_index",
                "transverse_kick_applied",
                "kick_kx_gev",
                "kick_ky_gev",
            ]
        )
        mwriter.writerows(metadata_rows)

    acceptance_fraction = accepted_events / total_generated if total_generated > 0 else float("nan")

    print(f"total_generated={total_generated}")
    print(f"total_accepted={accepted_events}")
    print(f"acceptance_fraction={acceptance_fraction:.6f}")
    if use_breit:
        print(f"breit_frame_rejections={breit_rejections}")
    else:
        print("breit_frame=off (lab momenta in CSV)")
    if kick_rng is not None:
        print(
            f"transverse_kicks_applied={kicks_applied} "
            f"kick_rolls_missing_pair={kicks_roll_no_pair} "
            f"(fraction={kick_fraction} |k|={kick_kt} GeV)"
        )
    else:
        print("transverse_kicks=off")
    print(f"output_event_record={out_event_record}")
    print(f"output_metadata={out_metadata}")

    n_files, capped = count_files_under(out_root)
    manifest_path = write_run_manifest(
        run_label="generate_dis_isr_parton_dataset",
        script_name="scripts/analysis/generate_dis_isr_parton_dataset.py",
        top_level_dirs_written=[str(out_root)],
        approximate_files_created=n_files,
        approximate_files_capped=capped,
        extra={
            "n_accepted_events": accepted_events,
            "n_generated_total": total_generated,
        },
    )
    print(f"run_manifest={manifest_path}")


if __name__ == "__main__":
    main()


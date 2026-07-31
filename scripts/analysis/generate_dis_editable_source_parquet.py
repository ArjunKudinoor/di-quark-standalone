#!/usr/bin/env python3
"""
Generate editable DIS source events (full parton-level event record, pre-hadronization) as
sharded Parquet — same physics/selection as ``generate_dis_isr_parton_dataset.py``, without
a giant monolithic CSV or per-event files.

Default output parent: ``~/Data/dis_isr_editable_source_100k`` → ``editable_source_v1/``.

Kinematic acceptance uses physics ``Q² = −q²`` and ``x = Q²/(2P·q)`` from electron
four-vectors (via ``_dis_kinematics_from_event``), not Pythia ``info.Q2Fac()`` / ``info.x2()``.

Compatibility: use ``editable_source_parquet.load_editable_event_dataframe`` for the same
per-event ``DataFrame`` shape as ``dis_isr_full_event_record.csv``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import pythia8
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pythia8 import failed: {exc}") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diquark.analyze_events_raw import FLIP_Z_PTREL, flip_z  # noqa: E402

import generate_dis_isr_parton_dataset as _gen  # noqa: E402

from editable_source_parquet import (  # noqa: E402
    BEAM_FLIP_LAB_COLUMNS,
    EDITABLE_SOURCE_V1_DIRNAME,
    EVENT_TABLE_COLUMNS,
    PARTICLE_CSV_COLUMN_ORDER,
    load_editable_event_dataframe,
    load_editable_metadata_row,
)

DATASET_ROOT = EDITABLE_SOURCE_V1_DIRNAME


def default_output_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_source_100k"


def build_pythia_source(
    seed: int,
    *,
    e_beam: float = 18.0,
    p_beam: float = 275.0,
    phase_space_q2_min: float = 16.0,
) -> pythia8.Pythia:
    """Match ``generate_dis_isr_parton_dataset.build_pythia`` with user ``Random:seed``."""
    p = pythia8.Pythia()
    p.readString("Beams:idA = 11")
    p.readString("Beams:idB = 2212")
    p.readString(f"Beams:eA = {float(e_beam)}")
    p.readString(f"Beams:eB = {float(p_beam)}")
    p.readString("Beams:frameType = 2")
    p.readString("WeakBosonExchange:ff2ff(t:gmZ) = on")
    p.readString(f"PhaseSpace:Q2Min = {float(phase_space_q2_min)}")
    p.readString("ProcessLevel:all = on")
    p.readString("PartonLevel:all = on")
    p.readString("PDF:lepton = off")
    p.readString("PartonLevel:ISR = off")
    p.readString("PartonLevel:FSR = off")
    p.readString("PartonLevel:MPI = off")
    p.readString("PartonLevel:Remnants = on")
    p.readString("HadronLevel:all = off")
    p.readString("Random:setSeed = on")
    p.readString(f"Random:seed = {int(seed)}")
    p.readString("Print:quiet = on")
    if not p.init():
        raise RuntimeError("PYTHIA init failed in build_pythia_source")
    return p


def _try_event_weight(pythia: pythia8.Pythia) -> float:
    try:
        info = pythia.infoPython()
        for name in ("weight", "getWeight"):
            if hasattr(info, name):
                w = getattr(info, name)
                return float(w() if callable(w) else w)
    except Exception:
        pass
    return float("nan")


def _dis_kinematics_from_event(ev: pythia8.Event) -> Optional[Dict[str, float]]:
    """
    Beam + virtual-photon kinematics (same construction as ``try_build_lt_from_event`` /
    ``build_LT``): Bjorken x from Q² and P·q, invariants Q, q_T, y, S=4E_e E_p, φ_q from q⊥.
    """
    beams = _gen.extract_beams_from_event(ev)
    if beams is None:
        return None
    e_in, e_sc, p_in = beams
    e_in_ev = flip_z(np.asarray(e_in, dtype=np.float64), FLIP_Z_PTREL)
    e_sc_ev = flip_z(np.asarray(e_sc, dtype=np.float64), FLIP_Z_PTREL)
    p_in_ev = flip_z(np.asarray(p_in, dtype=np.float64), FLIP_Z_PTREL)
    qmu = e_in_ev - e_sc_ev
    q0, q1, q2, q3 = float(qmu[0]), float(qmu[1]), float(qmu[2]), float(qmu[3])
    Q2 = -(q0 * q0 - q1 * q1 - q2 * q2 - q3 * q3)
    if Q2 <= 0.0:
        return None
    qT = math.hypot(q1, q2)
    phiq = math.atan2(q2, q1)
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
    if S <= 0.0 or x <= 0.0:
        return None
    y = Q2 / (S * x)
    Q = math.sqrt(Q2)
    return {"x": float(x), "Q": float(Q), "qT": float(qT), "y": float(y), "S": float(S), "phiq": float(phiq)}


def _flip_lab_beam_fourvectors(ev: pythia8.Event) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Incoming e, scattered e, incoming proton in the flipped lab frame used for LT (see ``_dis_kinematics_from_event``)."""
    beams = _gen.extract_beams_from_event(ev)
    if beams is None:
        return None
    e_in, e_sc, p_in = beams
    e_in_ev = flip_z(np.asarray(e_in, dtype=np.float64), FLIP_Z_PTREL)
    e_sc_ev = flip_z(np.asarray(e_sc, dtype=np.float64), FLIP_Z_PTREL)
    p_in_ev = flip_z(np.asarray(p_in, dtype=np.float64), FLIP_Z_PTREL)
    return e_in_ev, e_sc_ev, p_in_ev


def _beam_flip_lab_record(ev: pythia8.Event) -> Dict[str, float]:
    """12 floats for ``EVENT_TABLE_COLUMNS`` / ``BEAM_FLIP_LAB_COLUMNS`` (NaNs if beams missing)."""
    t = _flip_lab_beam_fourvectors(ev)
    if t is None:
        return {c: float("nan") for c in BEAM_FLIP_LAB_COLUMNS}
    ein, esc, pin = t
    vals = [
        float(ein[0]),
        float(ein[1]),
        float(ein[2]),
        float(ein[3]),
        float(esc[0]),
        float(esc[1]),
        float(esc[2]),
        float(esc[3]),
        float(pin[0]),
        float(pin[1]),
        float(pin[2]),
        float(pin[3]),
    ]
    return dict(zip(BEAM_FLIP_LAB_COLUMNS, vals))


def _particle_row_dicts_from_accepted_event(
    *,
    event_id: int,
    ev: pythia8.Event,
    LT: Optional[np.ndarray],
    use_breit: bool,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
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
        if not _gen.is_finite_particle(px, py, pz, E, m):
            continue
        if LT is not None:
            p4_lab = np.array([E, px, py, pz], dtype=np.float64)
            p4_lab = flip_z(p4_lab, FLIP_Z_PTREL)
            p4_b = LT @ p4_lab
            E = float(p4_b[0])
            px = float(p4_b[1])
            py = float(p4_b[2])
            pz = float(p4_b[3])
            m, pt, eta, phi = _gen.kinematics_from_p4(E, px, py, pz)
        else:
            m = float(p.m())
            pt = float(p.pT())
            eta = float(p.eta())
            phi = float(p.phi())
        if E <= 0:
            continue
        if not _gen.is_finite_particle(px, py, pz, E, m):
            continue
        rows.append(
            {
                "event_id": int(event_id),
                "particle_index": int(i),
                "pdg_id": int(pid),
                "status": int(status),
                "mother1": int(m1),
                "mother2": int(m2),
                "daughter1": int(d1),
                "daughter2": int(d2),
                "col": int(col),
                "acol": int(acol),
                "px": float(px),
                "py": float(py),
                "pz": float(pz),
                "E": float(E),
                "m": float(m),
                "pT": float(pt),
                "eta": float(eta),
                "phi": float(phi),
                "isFinal": int(is_final),
            }
        )
    return rows


def _event_rows_as_tuples(rows: List[Dict[str, Any]]) -> List[Tuple[Any, ...]]:
    """Convert to tuple rows indexed like ``generate_dis_isr_parton_dataset`` event_rows."""
    out: List[Tuple[Any, ...]] = []
    for r in rows:
        out.append(
            (
                r["event_id"],
                r["particle_index"],
                r["pdg_id"],
                r["status"],
                r["mother1"],
                r["mother2"],
                r["daughter1"],
                r["daughter2"],
                r["col"],
                r["acol"],
                r["px"],
                r["py"],
                r["pz"],
                r["E"],
                r["m"],
                r["pT"],
                r["eta"],
                r["phi"],
                r["isFinal"],
            )
        )
    return out


def _cast_particles_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "event_id": "int64",
            "particle_index": "int64",
            "pdg_id": "int32",
            "status": "int32",
            "mother1": "int32",
            "mother2": "int32",
            "daughter1": "int32",
            "daughter2": "int32",
            "col": "int32",
            "acol": "int32",
            "px": "float64",
            "py": "float64",
            "pz": "float64",
            "E": "float64",
            "m": "float64",
            "pT": "float64",
            "eta": "float64",
            "phi": "float64",
            "isFinal": "int32",
        }
    )


def _cast_events_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.astype(
        {
            "event_id": "int64",
            "Q2": "float64",
            "xB": "float64",
            "x": "float64",
            "Q": "float64",
            "qT": "float64",
            "y": "float64",
            "S": "float64",
            "phiq": "float64",
            "struck_incoming_index": "int32",
            "struck_outgoing_index": "int32",
            "transverse_kick_applied": "int32",
            "kick_kx_gev": "float64",
            "kick_ky_gev": "float64",
            "n_particles": "int32",
            "accepted": "int32",
            "weight": "float64",
        }
    )


def write_shard_pair(
    parent: Path,
    shard_idx: int,
    particles_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    base = parent / DATASET_ROOT
    pdir = base / "particles"
    edir = base / "events"
    pdir.mkdir(parents=True, exist_ok=True)
    edir.mkdir(parents=True, exist_ok=True)
    ppath = pdir / f"shard_{shard_idx:06d}.parquet"
    epath = edir / f"shard_{shard_idx:06d}.parquet"
    pdf = _cast_particles_df(particles_df)
    edf = _cast_events_df(events_df)
    pdf.to_parquet(ppath, index=False, compression="snappy", engine="pyarrow")
    edf.to_parquet(epath, index=False, compression="snappy", engine="pyarrow")
    fe = int(edf["event_id"].min())
    le = int(edf["event_id"].max())
    ne = int(len(edf))
    return (
        {
            "dataset": "particles",
            "shard_path": f"particles/shard_{shard_idx:06d}.parquet",
            "first_event_id": fe,
            "last_event_id": le,
            "n_events": ne,
            "n_rows": int(len(pdf)),
        },
        {
            "dataset": "events",
            "shard_path": f"events/shard_{shard_idx:06d}.parquet",
            "first_event_id": fe,
            "last_event_id": le,
            "n_events": ne,
            "n_rows": ne,
        },
    )


def _load_modify_helpers():
    path = _ANAL / "modify_dis_isr_parton_dataset.py"
    spec = importlib.util.spec_from_file_location("_dis_mod", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def run_post_generation_validation(parent: Path, events_per_shard: int) -> Dict[str, Any]:
    """Check loader + struck-quark resolution on event_id=0."""
    rep: Dict[str, Any] = {"ok": True, "errors": []}
    try:
        g = load_editable_event_dataframe(parent, 0, events_per_shard=events_per_shard)
        meta = load_editable_metadata_row(parent, 0, events_per_shard=events_per_shard)
        struck_in = int(meta["struck_incoming_index"])
        mod = _load_modify_helpers()
        reco = mod.find_outgoing_struck_quark_noisr(g, struck_in)
        if not reco.get("success"):
            rep["ok"] = False
            rep["errors"].append(f"find_outgoing_struck_quark_noisr: {reco.get('failure_reason')}")
        else:
            sel = int(reco["selected_row"]["particle_index"])
            exp = int(meta["struck_outgoing_index"])
            if exp >= 0 and sel != exp:
                rep["errors"].append(
                    f"struck_outgoing_index mismatch: metadata={exp} reco={sel} (informational)"
                )
        if not g["particle_index"].is_monotonic_increasing:
            rep["ok"] = False
            rep["errors"].append("particle_index not sorted")
        if list(g.columns) != PARTICLE_CSV_COLUMN_ORDER:
            rep["ok"] = False
            rep["errors"].append("column order mismatch vs CSV convention")
    except Exception as exc:
        rep["ok"] = False
        rep["errors"].append(f"{type(exc).__name__}: {exc}")
    return rep


def run_kinematic_validation(
    parent: Path,
    *,
    x_strict_above: Optional[float],
    q2_strict_above: Optional[float],
    q2_strict_below: Optional[float],
    require_struck_u: bool,
) -> Dict[str, Any]:
    """
    Scan all events + particles shards: every accepted row must satisfy strict DIS cuts
    (xB, Q2) and optionally pdg at struck_incoming_index == 2 (u).
    """
    rep: Dict[str, Any] = {
        "ok": True,
        "n_event_rows": 0,
        "n_fail_x": 0,
        "n_fail_q2_lo": 0,
        "n_fail_q2_hi": 0,
        "n_fail_pdg": 0,
        "first_bad": None,
    }
    base = parent / DATASET_ROOT
    ev_paths = sorted((base / "events").glob("shard_*.parquet"))
    pp_paths = sorted((base / "particles").glob("shard_*.parquet"))
    if not ev_paths:
        rep["ok"] = False
        rep["first_bad"] = "no event shards"
        return rep

    for ep in ev_paths:
        edf = pd.read_parquet(ep)
        rep["n_event_rows"] += int(len(edf))
        shard_idx = int(ep.stem.split("_")[-1])
        pp = base / "particles" / f"shard_{shard_idx:06d}.parquet"
        if not pp.is_file():
            rep["ok"] = False
            rep["first_bad"] = f"missing particles for {ep.name}"
            return rep
        pdf = pd.read_parquet(pp)

        for _, row in edf.iterrows():
            eid = int(row["event_id"])
            q2 = float(row["Q2"])
            xb = float(row["xB"])
            inc_i = int(row["struck_incoming_index"])
            if x_strict_above is not None and not (xb > x_strict_above):
                rep["n_fail_x"] += 1
                if rep["first_bad"] is None:
                    rep["first_bad"] = {
                        "reason": "xB",
                        "event_id": eid,
                        "xB": xb,
                        "limit": x_strict_above,
                    }
            if q2_strict_above is not None and not (q2 > q2_strict_above):
                rep["n_fail_q2_lo"] += 1
                if rep["first_bad"] is None:
                    rep["first_bad"] = {
                        "reason": "Q2_lower",
                        "event_id": eid,
                        "Q2": q2,
                        "limit": q2_strict_above,
                    }
            if q2_strict_below is not None and not (q2 < q2_strict_below):
                rep["n_fail_q2_hi"] += 1
                if rep["first_bad"] is None:
                    rep["first_bad"] = {
                        "reason": "Q2_upper",
                        "event_id": eid,
                        "Q2": q2,
                        "limit": q2_strict_below,
                    }
            if require_struck_u:
                sub = pdf[(pdf["event_id"] == eid) & (pdf["particle_index"] == inc_i)]
                pdg_ok = len(sub) == 1 and int(sub.iloc[0]["pdg_id"]) == 2
                if not pdg_ok:
                    rep["n_fail_pdg"] += 1
                    if rep["first_bad"] is None:
                        rep["first_bad"] = {
                            "reason": "struck_u",
                            "event_id": eid,
                            "inc_idx": inc_i,
                            "pdg": int(sub.iloc[0]["pdg_id"]) if len(sub) == 1 else None,
                        }

    n_bad = (
        rep["n_fail_x"]
        + rep["n_fail_q2_lo"]
        + rep["n_fail_q2_hi"]
        + rep["n_fail_pdg"]
    )
    if n_bad > 0:
        rep["ok"] = False
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description="Editable DIS source → sharded Parquet (full record).")
    ap.add_argument(
        "--kinematics-preset",
        choices=("none", "e5_ep41_x005_q2to4"),
        default="none",
        help=(
            "'e5_ep41_x005_q2to4' sets beams 5×41 GeV, PhaseSpace:Q2Min=4 GeV², and strict sampling "
            "xB > 0.05 with 4 < Q² < 16 GeV² (equivalently 2 < Q < 4 GeV). Overrides conflicting "
            "beam/Q²/x flags below."
        ),
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=f"Parent directory (creates {DATASET_ROOT}/). Default: {default_output_parent()}",
    )
    ap.add_argument("--n-accepted", type=int, default=100_000, help="Target accepted events.")
    ap.add_argument(
        "--event-id-offset",
        type=int,
        default=0,
        help=(
            "First accepted event_id (then +1 each acceptance). Use disjoint ranges for parallel "
            "workers on the same --output-dir (e.g. worker i uses offset i * n_accepted per worker)."
        ),
    )
    ap.add_argument(
        "--output-tag",
        type=str,
        default="",
        help=(
            "If set, write manifest.parquet as manifest_{tag}.parquet and run_summary as "
            "run_summary_{tag}.json (avoids races when multiple workers share one output parent)."
        ),
    )
    ap.add_argument("--seed", type=int, default=12345, help="PYTHIA Random:seed.")
    ap.add_argument(
        "--beam-e",
        type=float,
        default=18.0,
        help="Beams:eA (electron energy in GeV).",
    )
    ap.add_argument(
        "--beam-p",
        type=float,
        default=275.0,
        help="Beams:eB (proton energy in GeV).",
    )
    ap.add_argument(
        "--phase-space-q2-min",
        type=float,
        default=16.0,
        help="PYTHIA PhaseSpace:Q2Min (GeV^2). Use 4.0 to allow Q>2 GeV; tighten with --accept-q2-*.",
    )
    ap.add_argument(
        "--accept-x-min",
        type=float,
        default=None,
        help="If set, accept only events with xB strictly above this (Bjorken x from PYTHIA info).",
    )
    ap.add_argument(
        "--accept-q2-min",
        type=float,
        default=None,
        help="If set, accept only Q2 (GeV^2) strictly above this (matches strict 4 < Q2 when 4.0).",
    )
    ap.add_argument(
        "--accept-q2-max",
        type=float,
        default=None,
        help="If set, accept only Q2 (GeV^2) strictly below this (e.g. 25 for Q<5 GeV).",
    )
    ap.add_argument(
        "--events-per-shard",
        type=int,
        default=10_000,
        help="Accepted events per Parquet shard (default 10000).",
    )
    ap.add_argument(
        "--no-breit-frame",
        action="store_true",
        help="Store lab-frame momenta (same flag semantics as generate_dis_isr_parton_dataset).",
    )
    ap.add_argument(
        "--kick-fraction",
        type=float,
        default=0.0,
        help="Bernoulli fraction for in-generator transverse kick (default 0 for clean editable pool).",
    )
    ap.add_argument(
        "--kick-kt-gev",
        type=float,
        default=0.4,
        help="|k| for paired kick when kick-fraction > 0.",
    )
    ap.add_argument(
        "--kick-seed",
        type=int,
        default=98765,
        help="RNG seed for kick subsampling (independent of PYTHIA seed).",
    )
    ap.add_argument(
        "--auto-validate-max",
        type=int,
        default=10_000,
        help="If n_accepted <= this, run loader/struck-quark validation after generation.",
    )
    args = ap.parse_args()

    kinematics_preset = str(args.kinematics_preset)
    if kinematics_preset == "e5_ep41_x005_q2to4":
        args.beam_e = 5.0
        args.beam_p = 41.0
        args.phase_space_q2_min = 4.0
        args.accept_x_min = 0.05
        args.accept_q2_min = 4.0
        args.accept_q2_max = 16.0

    parent = (args.output_dir or default_output_parent()).resolve()
    n_target = int(args.n_accepted)
    events_per_shard = max(1, int(args.events_per_shard))
    beam_e = float(args.beam_e)
    beam_p = float(args.beam_p)
    phase_space_q2_min = float(args.phase_space_q2_min)
    accept_x_min = args.accept_x_min
    accept_q2_min = args.accept_q2_min
    accept_q2_max = args.accept_q2_max
    use_breit = not bool(args.no_breit_frame)
    kick_fraction = min(1.0, max(0.0, float(args.kick_fraction)))
    kick_kt = max(0.0, float(args.kick_kt_gev))
    kick_rng = (
        random.Random(int(args.kick_seed)) if kick_fraction > 0 and kick_kt > 0 else None
    )

    event_id_offset = int(args.event_id_offset)
    output_tag = str(args.output_tag).strip()
    out_suffix = f"_{output_tag}" if output_tag else ""

    base = parent / DATASET_ROOT
    (base / "particles").mkdir(parents=True, exist_ok=True)
    (base / "events").mkdir(parents=True, exist_ok=True)

    pythia = build_pythia_source(
        int(args.seed),
        e_beam=beam_e,
        p_beam=beam_p,
        phase_space_q2_min=phase_space_q2_min,
    )
    ev = pythia.event

    particle_buf: List[Dict[str, Any]] = []
    event_buf: List[Dict[str, Any]] = []
    manifest_rows: List[Dict[str, Any]] = []

    total_generated = 0
    accepted = 0
    breit_rejections = 0
    kinematic_rejections = 0
    kicks_applied = 0
    kicks_roll_no_pair = 0
    t0 = time.perf_counter()

    def flush_shard() -> None:
        nonlocal particle_buf, event_buf, manifest_rows
        if not event_buf:
            return
        first_eid = int(event_buf[0]["event_id"])
        shard_idx = first_eid // events_per_shard
        pdf = pd.DataFrame(particle_buf)
        edf = pd.DataFrame(event_buf)
        pr, er = write_shard_pair(parent, shard_idx, pdf, edf)
        manifest_rows.append(pr)
        manifest_rows.append(er)
        particle_buf = []
        event_buf = []

    while accepted < n_target:
        if not pythia.next():
            continue
        total_generated += 1

        inc_idx = _gen.pick_incoming_quark_index(ev)
        if inc_idx is None:
            continue
        # Struck up quark only (not anti-up: id -2).
        if int(ev[inc_idx].id()) != 2:
            continue

        LT: Optional[np.ndarray] = None
        if use_breit:
            LT = _gen.try_build_lt_from_event(ev)
            if LT is None:
                breit_rejections += 1
                continue

        kin = _dis_kinematics_from_event(ev)
        if kin is None:
            kinematic_rejections += 1
            continue

        q2 = float(kin["Q"]) ** 2
        xb = float(kin["x"])

        if not math.isfinite(q2) or not math.isfinite(xb):
            kinematic_rejections += 1
            continue
        if accept_x_min is not None and not (xb > float(accept_x_min)):
            kinematic_rejections += 1
            continue
        if accept_q2_min is not None and not (q2 > float(accept_q2_min)):
            kinematic_rejections += 1
            continue
        if accept_q2_max is not None and not (q2 < float(accept_q2_max)):
            kinematic_rejections += 1
            continue

        event_id = event_id_offset + accepted
        struck_out_idx = _gen.identify_struck_outgoing_quark_index(ev, inc_idx)

        row_dicts = _particle_row_dicts_from_accepted_event(
            event_id=event_id, ev=ev, LT=LT, use_breit=use_breit
        )
        event_tuples = _event_rows_as_tuples(row_dicts)

        kick_applied_flag = 0
        kick_kx_out = 0.0
        kick_ky_out = 0.0
        if kick_rng is not None and kick_rng.random() < kick_fraction:
            li_dq = _gen.find_event_row_list_index_for_diquark_kick_partner(event_tuples)
            li_sq = (
                _gen.find_event_row_list_index_for_particle_index(event_tuples, struck_out_idx)
                if struck_out_idx >= 0
                else None
            )
            if li_dq is not None and li_sq is not None and li_dq != li_sq:
                phi_k = kick_rng.uniform(0.0, 2.0 * math.pi)
                kx = kick_kt * math.cos(phi_k)
                ky = kick_kt * math.sin(phi_k)
                _gen.apply_balanced_transverse_kick_two_rows(event_tuples, li_sq, li_dq, kx, ky)
                kick_applied_flag = 1
                kick_kx_out = kx
                kick_ky_out = ky
                kicks_applied += 1
                row_dicts = [
                    {
                        "event_id": int(t[0]),
                        "particle_index": int(t[1]),
                        "pdg_id": int(t[2]),
                        "status": int(t[3]),
                        "mother1": int(t[4]),
                        "mother2": int(t[5]),
                        "daughter1": int(t[6]),
                        "daughter2": int(t[7]),
                        "col": int(t[8]),
                        "acol": int(t[9]),
                        "px": float(t[10]),
                        "py": float(t[11]),
                        "pz": float(t[12]),
                        "E": float(t[13]),
                        "m": float(t[14]),
                        "pT": float(t[15]),
                        "eta": float(t[16]),
                        "phi": float(t[17]),
                        "isFinal": int(t[18]),
                    }
                    for t in event_tuples
                ]
            else:
                kicks_roll_no_pair += 1

        w = _try_event_weight(pythia)

        particle_buf.extend(row_dicts)
        event_buf.append(
            {
                "event_id": int(event_id),
                "Q2": float(q2),
                "xB": float(xb),
                "x": float(kin["x"]),
                "Q": float(kin["Q"]),
                "qT": float(kin["qT"]),
                "y": float(kin["y"]),
                "S": float(kin["S"]),
                "phiq": float(kin["phiq"]),
                "struck_incoming_index": int(inc_idx),
                "struck_outgoing_index": int(struck_out_idx),
                "transverse_kick_applied": int(kick_applied_flag),
                "kick_kx_gev": float(kick_kx_out),
                "kick_ky_gev": float(kick_ky_out),
                "n_particles": int(len(row_dicts)),
                "accepted": 1,
                "weight": float(w),
                **_beam_flip_lab_record(ev),
            }
        )
        accepted += 1

        if len(event_buf) >= events_per_shard:
            flush_shard()

        if accepted % 10_000 == 0:
            dt = time.perf_counter() - t0
            print(
                f"accepted={accepted}/{n_target}  rate={accepted / max(dt, 1e-9):.1f} ev/s  "
                f"breit_rej={breit_rejections}  gen={total_generated}",
                flush=True,
            )

    flush_shard()

    man_df = pd.DataFrame(manifest_rows)
    man_path = base / f"manifest{out_suffix}.parquet"
    man_df.to_parquet(man_path, index=False, compression="snappy", engine="pyarrow")

    elapsed = time.perf_counter() - t0
    all_p = sorted((base / "particles").glob("shard_*.parquet"))
    total_bytes = sum(p.stat().st_size for p in all_p)
    total_bytes += sum(p.stat().st_size for p in sorted((base / "events").glob("shard_*.parquet")))
    total_bytes += man_path.stat().st_size

    q_min_gev = math.sqrt(float(accept_q2_min)) if accept_q2_min is not None else None
    q_max_gev = math.sqrt(float(accept_q2_max)) if accept_q2_max is not None else None
    summary = {
        "output_parent": str(parent),
        "dataset_root": str(base),
        "kinematics_preset": kinematics_preset,
        "event_id_offset": int(event_id_offset),
        "output_tag": output_tag,
        "manifest_path": str(man_path.name),
        "n_accepted": accepted,
        "n_generated_tried": total_generated,
        "breit_rejections": breit_rejections,
        "kinematic_rejections": kinematic_rejections,
        "events_per_shard": events_per_shard,
        "n_particle_shards": len(all_p),
        "n_manifest_rows": int(len(manifest_rows)),
        "total_disk_bytes": int(total_bytes),
        "elapsed_s": float(elapsed),
        "acceptance_fraction": float(accepted / total_generated) if total_generated else 0.0,
        "use_breit_frame": use_breit,
        "kick_fraction": kick_fraction,
        "kick_kt_gev": kick_kt,
        "kicks_applied": kicks_applied,
        "kicks_roll_no_pair": kicks_roll_no_pair,
        "pythia_seed": int(args.seed),
        "Ee": beam_e,
        "Ep": beam_p,
        "phase_space_Q2Min": phase_space_q2_min,
        "x_min": accept_x_min,
        "Q_min": q_min_gev,
        "Q_max": q_max_gev,
        "Q2_min": accept_q2_min,
        "Q2_max": accept_q2_max,
        "particle_columns": PARTICLE_CSV_COLUMN_ORDER,
        "event_columns": EVENT_TABLE_COLUMNS,
    }
    summary_path = base / f"run_summary{out_suffix}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if n_target <= int(args.auto_validate_max):
        vrep = run_post_generation_validation(parent, events_per_shard)
        print("validation:", json.dumps(vrep, indent=2))
        if not vrep["ok"]:
            raise SystemExit(1)
        if (
            accept_x_min is not None
            or accept_q2_min is not None
            or accept_q2_max is not None
        ):
            krep = run_kinematic_validation(
                parent,
                x_strict_above=float(accept_x_min) if accept_x_min is not None else None,
                q2_strict_above=float(accept_q2_min) if accept_q2_min is not None else None,
                q2_strict_below=float(accept_q2_max) if accept_q2_max is not None else None,
                require_struck_u=True,
            )
            print("kinematic_validation:", json.dumps(krep, indent=2))
            if not krep["ok"]:
                raise SystemExit(1)


if __name__ == "__main__":
    main()

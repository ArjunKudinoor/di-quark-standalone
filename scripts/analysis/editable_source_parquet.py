"""
Sharded Parquet layout for editable DIS source events (pre-hadronization full record).

Use :func:`load_editable_event_dataframe` to obtain a per-event ``DataFrame`` with the same
column names and semantics as ``dis_isr_full_event_record.csv`` for
``split_dis_sample_diquark_kick.py`` / ``modify_dis_isr_parton_dataset`` workflows.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import pandas as pd

EDITABLE_SOURCE_V1_DIRNAME = "editable_source_v1"

# Same column order as ``generate_dis_isr_parton_dataset`` CSV header (full event record).
PARTICLE_CSV_COLUMN_ORDER: List[str] = [
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

# Flipped lab beam four-vectors (E, px, py, pz), GeV — same convention as
# ``analyze_events_raw.FLIP_Z_PTREL`` in the DIS / reinjection pipelines. Used to rebuild
# lab→Breit ``LT`` when particle momenta are stored in Breit (η acceptance in collider frame).
BEAM_FLIP_LAB_COLUMNS: List[str] = [
    "ein_fl_E",
    "ein_fl_px",
    "ein_fl_py",
    "ein_fl_pz",
    "esc_fl_E",
    "esc_fl_px",
    "esc_fl_py",
    "esc_fl_pz",
    "pin_fl_E",
    "pin_fl_px",
    "pin_fl_py",
    "pin_fl_pz",
]

# Same as ``dis_isr_event_metadata.csv`` from the generator (plus accepted flag for Parquet).
_EVENT_TABLE_BASE: List[str] = [
    "event_id",
    "Q2",
    "xB",
    "x",
    "Q",
    "qT",
    "y",
    "S",
    "phiq",
    "struck_incoming_index",
    "struck_outgoing_index",
    "transverse_kick_applied",
    "kick_kx_gev",
    "kick_ky_gev",
    "n_particles",
    "accepted",
    "weight",
]

EVENT_TABLE_COLUMNS: List[str] = _EVENT_TABLE_BASE + BEAM_FLIP_LAB_COLUMNS

METADATA_CSV_COLUMNS: List[str] = [
    "event_id",
    "Q2",
    "xB",
    "struck_incoming_index",
    "struck_outgoing_index",
    "transverse_kick_applied",
    "kick_kx_gev",
    "kick_ky_gev",
]


def default_events_per_shard_from_summary(parent: Path) -> int:
    """Read ``run_summary.json`` if present; else default 10_000."""
    p = parent / EDITABLE_SOURCE_V1_DIRNAME / "run_summary.json"
    if p.is_file():
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
            return int(d.get("events_per_shard", 10_000))
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return 10_000


def _particles_path(parent: Path, shard_idx: int) -> Path:
    return parent / EDITABLE_SOURCE_V1_DIRNAME / "particles" / f"shard_{shard_idx:06d}.parquet"


def _events_path(parent: Path, shard_idx: int) -> Path:
    return parent / EDITABLE_SOURCE_V1_DIRNAME / "events" / f"shard_{shard_idx:06d}.parquet"


def shard_index_for_event(event_id: int, events_per_shard: int) -> int:
    if events_per_shard <= 0:
        raise ValueError("events_per_shard must be positive")
    return int(event_id) // events_per_shard


def iter_particle_shard_paths(output_parent: Path) -> Iterator[Path]:
    """Sorted ``particles/shard_*.parquet`` paths (for adapting batch pipelines)."""
    base = output_parent.resolve() / EDITABLE_SOURCE_V1_DIRNAME / "particles"
    if base.is_dir():
        yield from sorted(base.glob("shard_*.parquet"))


def load_editable_event_dataframe(
    output_parent: Path,
    event_id: int,
    *,
    events_per_shard: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load one event as a ``DataFrame`` matching the old full-event CSV row layout
    (columns and ``particle_index`` = PYTHIA listing index).
    """
    parent = output_parent.resolve()
    eps = int(events_per_shard) if events_per_shard is not None else default_events_per_shard_from_summary(parent)
    k = shard_index_for_event(event_id, eps)
    path = _particles_path(parent, k)
    if not path.is_file():
        raise FileNotFoundError(f"missing particles shard: {path}")
    df = pd.read_parquet(path)
    g = df[df["event_id"] == int(event_id)].copy()
    if g.empty:
        raise KeyError(f"no particles for event_id={event_id} in {path}")
    g = g.sort_values("particle_index").reset_index(drop=True)
    missing = [c for c in PARTICLE_CSV_COLUMN_ORDER if c not in g.columns]
    if missing:
        raise ValueError(f"Parquet missing columns: {missing}")
    return g[PARTICLE_CSV_COLUMN_ORDER].copy()


def load_editable_metadata_row(
    output_parent: Path,
    event_id: int,
    *,
    events_per_shard: Optional[int] = None,
) -> pd.Series:
    """One metadata row as a Series (includes n_particles, accepted, weight)."""
    parent = output_parent.resolve()
    eps = int(events_per_shard) if events_per_shard is not None else default_events_per_shard_from_summary(parent)
    k = shard_index_for_event(event_id, eps)
    path = _events_path(parent, k)
    if not path.is_file():
        raise FileNotFoundError(f"missing events shard: {path}")
    edf = pd.read_parquet(path)
    rows = edf[edf["event_id"] == int(event_id)]
    if rows.empty:
        raise KeyError(f"no event row for event_id={event_id} in {path}")
    return rows.iloc[0]


def metadata_row_for_split_script(meta_row: pd.Series) -> Dict[str, Any]:
    """Fields expected when building a metadata map for ``split_dis_sample_diquark_kick``."""
    return {
        "event_id": int(meta_row["event_id"]),
        "Q2": float(meta_row["Q2"]),
        "xB": float(meta_row["xB"]),
        "struck_incoming_index": int(meta_row["struck_incoming_index"]),
        "struck_outgoing_index": int(meta_row["struck_outgoing_index"]),
        "transverse_kick_applied": int(meta_row.get("transverse_kick_applied", 0)),
        "kick_kx_gev": float(meta_row.get("kick_kx_gev", 0.0)),
        "kick_ky_gev": float(meta_row.get("kick_ky_gev", 0.0)),
    }


def load_metadata_slice_as_dataframe(
    output_parent: Path,
    event_ids: List[int],
    *,
    events_per_shard: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a small ``DataFrame`` in ``dis_isr_event_metadata.csv`` column order for a list
    of ``event_id`` (loads only touched event shards).
    """
    parent = output_parent.resolve()
    eps = int(events_per_shard) if events_per_shard is not None else default_events_per_shard_from_summary(parent)
    rows: List[Dict[str, Any]] = []
    for eid in sorted(set(int(x) for x in event_ids)):
        s = load_editable_metadata_row(parent, eid, events_per_shard=eps)
        rows.append(metadata_row_for_split_script(s))
    return pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)

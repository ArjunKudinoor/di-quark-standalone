#!/usr/bin/env python3
"""
Apply diquark split + px kick to **every** event in ``editable_source_v1`` (no 90/10 inside
the 100k pool). Intended workflow:

- **900k** background = unchanged control (separate dataset).
- **100k** editable source → **this script** → **100k** rows in ``altered_100k_parquet/``,
  each processed with ``event_in_alter_pool=True`` (physics fallbacks still possible).

Output: ``<out-parent>/altered_100k_parquet/`` with sharded Parquet + ``tqdm`` progress.

**Hard topology:** every altered event uses channel **C**,
``(ud) -> [d] + (us) + sbar`` (PYTHIA ``su_0`` = 3201 + ``sbar``).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "tqdm is required: pip install tqdm (see requirements.txt)"
    ) from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC = _PROJECT_ROOT / "src"
_ANAL = Path(__file__).resolve().parent
for _p in (_SRC, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from editable_source_parquet import (  # noqa: E402
    BEAM_FLIP_LAB_COLUMNS,
    EDITABLE_SOURCE_V1_DIRNAME,
    EVENT_TABLE_COLUMNS,
    PARTICLE_CSV_COLUMN_ORDER,
)

OUTPUT_SUBDIR = "altered_100k_parquet"


def default_editable_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_source_100k"


def default_out_parent() -> Path:
    return Path.home() / "Data" / "dis_isr_editable_altered_100k"


def _read_editable_run_summary(editable_v1: Path) -> int:
    p = editable_v1 / "run_summary.json"
    if not p.is_file():
        raise FileNotFoundError(f"missing {p}")
    d = json.loads(p.read_text(encoding="utf-8"))
    return int(d["n_accepted"])


def _iter_input_shard_pairs(editable_v1: Path) -> List[Tuple[Path, Path]]:
    pdir = editable_v1 / "particles"
    pairs: List[Tuple[Path, Path]] = []
    for pp in sorted(pdir.glob("shard_*.parquet")):
        ep = editable_v1 / "events" / f"{pp.stem}.parquet"
        if not ep.is_file():
            raise FileNotFoundError(f"missing matching events shard for {pp.name}")
        pairs.append((pp, ep))
    if not pairs:
        raise FileNotFoundError(f"no particle shards under {pdir}")
    return pairs


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
    d: Dict[str, str] = {
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
    for c in BEAM_FLIP_LAB_COLUMNS:
        d[c] = "float64"
    return df.astype(d)


def _flatten_altered_meta_success(event_id: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    npd = meta.get("new_particle_pdgs") or {}
    mode = str(meta.get("mode", "diquark_kick"))
    row: Dict[str, Any] = {
        "event_id": int(event_id),
        "meta_json": json.dumps(meta, sort_keys=True),
        "split_mode": mode,
        "alteration_succeeded": 1,
        "failure_reason": "",
    }
    if mode == "breit_px_kick_only":
        row["split_channel"] = ""
        row["channel_description"] = ""
        row["struck_quark_index"] = int(meta.get("struck_quark_index", -1))
        row["kick_partner_particle_index"] = int(meta.get("kick_partner_particle_index", -1))
        row["kick_delta_px_gev"] = float(meta.get("kick_delta_px_gev", float("nan")))
        row["original_diquark_index"] = -1
        row["stripped_quark_index"] = -1
        row["created_antiquark_index"] = -1
        row["daughter_diquark_index"] = -1
        row["color_tag_A"] = -1
        row["color_tag_B"] = -1
        row["pdg_stripped"] = 0
        row["pdg_antiquark"] = 0
        row["pdg_daughter_diquark"] = 0
        row["kick_theta_rad"] = float("nan")
        row["kick_magnitude_gev"] = float("nan")
        row["x_struck_used"] = float("nan")
    else:
        row["split_channel"] = str(meta.get("split_channel", ""))
        row["channel_description"] = str(meta.get("channel_description", ""))
        row["struck_quark_index"] = int(meta.get("struck_quark_index", -1))
        row["kick_partner_particle_index"] = -1
        row["kick_delta_px_gev"] = float(meta.get("kick_delta_px_gev", float("nan")))
        row["original_diquark_index"] = int(meta.get("original_diquark_index", -1))
        row["stripped_quark_index"] = int(meta.get("stripped_quark_index", -1))
        row["created_antiquark_index"] = int(meta.get("created_antiquark_index", -1))
        row["daughter_diquark_index"] = int(meta.get("daughter_diquark_index", -1))
        row["color_tag_A"] = int(meta.get("color_tag_A", -1))
        row["color_tag_B"] = int(meta.get("color_tag_B", -1))
        row["pdg_stripped"] = int(npd.get("stripped", 0))
        row["pdg_antiquark"] = int(npd.get("antiquark", 0))
        row["pdg_daughter_diquark"] = int(npd.get("daughter_diquark", 0))
        row["kick_theta_rad"] = float(meta.get("kick_theta_rad", float("nan")))
        row["kick_magnitude_gev"] = float(meta.get("kick_magnitude_gev", float("nan")))
        row["x_struck_used"] = float(meta.get("x_struck_used", float("nan")))
    return row


def _failure_reason_from_delta(delta: Dict[str, int]) -> str:
    if delta.get("written_altered"):
        return ""
    if delta.get("fallback_unchanged_struck_fail"):
        return "struck_unresolved"
    if delta.get("fallback_unchanged_no_ud_diquark"):
        return "no_ud_diquark"
    if delta.get("fallback_unchanged_validation_fail"):
        return "validation_fail"
    return "unknown"


def _flatten_altered_meta_failure(event_id: int, reason: str) -> Dict[str, Any]:
    return {
        "event_id": int(event_id),
        "meta_json": "",
        "split_mode": "",
        "split_channel": "",
        "channel_description": "",
        "struck_quark_index": -1,
        "kick_partner_particle_index": -1,
        "kick_delta_px_gev": float("nan"),
        "original_diquark_index": -1,
        "stripped_quark_index": -1,
        "created_antiquark_index": -1,
        "daughter_diquark_index": -1,
        "color_tag_A": -1,
        "color_tag_B": -1,
        "pdg_stripped": 0,
        "pdg_antiquark": 0,
        "pdg_daughter_diquark": 0,
        "kick_theta_rad": float("nan"),
        "kick_magnitude_gev": float("nan"),
        "x_struck_used": float("nan"),
        "alteration_succeeded": 0,
        "failure_reason": reason,
    }


ALTERED_META_COLUMNS = list(_flatten_altered_meta_failure(0, "").keys())


class _AlteredAllWriter:
    def __init__(self, out_parent: Path, events_per_shard: int) -> None:
        self.root = out_parent / OUTPUT_SUBDIR
        (self.root / "particles").mkdir(parents=True, exist_ok=True)
        (self.root / "events").mkdir(parents=True, exist_ok=True)
        (self.root / "altered_metadata").mkdir(parents=True, exist_ok=True)
        self.events_per_shard = max(1, int(events_per_shard))
        self.manifest_rows: List[Dict[str, Any]] = []
        self._shard_idx = 0
        self._p_buf: List[pd.DataFrame] = []
        self._e_buf: List[Dict[str, Any]] = []
        self._m_buf: List[Dict[str, Any]] = []
        self._ev_count = 0

    def _flush(self) -> None:
        if not self._e_buf:
            return
        pdf = pd.concat(self._p_buf, ignore_index=True)
        edf = pd.DataFrame(self._e_buf)
        mdf = pd.DataFrame(self._m_buf)
        for col in ALTERED_META_COLUMNS:
            if col not in mdf.columns:
                mdf[col] = pd.NA
        mdf = mdf[ALTERED_META_COLUMNS]
        pdf = _cast_particles_df(pdf[PARTICLE_CSV_COLUMN_ORDER])
        edf = _cast_events_df(edf[EVENT_TABLE_COLUMNS])
        si = self._shard_idx
        pdf.to_parquet(
            self.root / "particles" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        edf.to_parquet(
            self.root / "events" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        mdf.to_parquet(
            self.root / "altered_metadata" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        fe = int(edf["event_id"].min())
        le = int(edf["event_id"].max())
        ne = int(len(edf))
        for dataset, nrows in (
            ("particles", int(len(pdf))),
            ("events", ne),
            ("altered_metadata", int(len(mdf))),
        ):
            subdir = "altered_metadata" if dataset == "altered_metadata" else dataset
            self.manifest_rows.append(
                {
                    "dataset": dataset,
                    "shard_path": f"{subdir}/shard_{si:06d}.parquet",
                    "first_event_id": fe,
                    "last_event_id": le,
                    "n_events": ne,
                    "n_rows": nrows,
                }
            )
        self._shard_idx += 1
        self._p_buf = []
        self._e_buf = []
        self._m_buf = []
        self._ev_count = 0

    def append(
        self,
        particles: pd.DataFrame,
        event_row: Dict[str, Any],
        meta_row: Dict[str, Any],
    ) -> None:
        self._p_buf.append(particles[PARTICLE_CSV_COLUMN_ORDER].copy())
        self._e_buf.append(event_row)
        self._m_buf.append(meta_row)
        self._ev_count += 1
        if self._ev_count >= self.events_per_shard:
            self._flush()

    def finalize(self) -> None:
        self._flush()
        man = pd.DataFrame(self.manifest_rows)
        man.to_parquet(
            self.root / "manifest.parquet", index=False, compression="snappy", engine="pyarrow"
        )


def _event_row_dict(md_row: Optional[pd.Series], eid: int, n_part: int) -> Dict[str, Any]:
    if md_row is None:
        raise ValueError(f"missing events-table row for event_id={eid}")
    d: Dict[str, Any] = {"event_id": int(eid), "n_particles": int(n_part)}
    for c in EVENT_TABLE_COLUMNS:
        if c in ("event_id", "n_particles"):
            continue
        if c not in md_row.index:
            raise ValueError(f"events shard missing column {c!r} for event_id={eid}")
        d[c] = md_row[c]
    return d


def _split_list(xs: List[Any], n_parts: int) -> List[List[Any]]:
    """Split ``xs`` into up to ``n_parts`` contiguous chunks (last chunk may be smaller)."""
    if not xs:
        return []
    n = max(1, min(int(n_parts), len(xs)))
    k, m = divmod(len(xs), n)
    out: List[List[Any]] = []
    start = 0
    for i in range(n):
        size = k + (1 if i < m else 0)
        out.append(xs[start : start + size])
        start += size
    return out


def _alter_shard_worker(payload: dict) -> dict:
    """
    Process one batch of input shard pairs. Writes output shard pairs directly
    using the input shard index as the output shard index (one-to-one).
    Returns the local stats dict for merging in the parent.
    """
    import random as _random

    editable_v1 = Path(payload["editable_v1"])
    out_root = Path(payload["out_root"])
    shard_names: List[str] = list(payload["shard_names"])
    id_allowed: Set[int] = set(int(x) for x in payload["id_allowed"])
    seed: int = int(payload["seed"])
    delta_px: float = float(payload["delta_px"])
    mode: str = str(payload["mode"])
    surgery_version: str = str(payload["surgery_version"])

    if surgery_version == "v2":
        from split_dis_sample_diquark_kick_v2 import process_one_event_split  # noqa: WPS433
    else:
        from split_dis_sample_diquark_kick import process_one_event_split  # noqa: WPS433

    rng = _random.Random(int(seed))

    (out_root / "particles").mkdir(parents=True, exist_ok=True)
    (out_root / "events").mkdir(parents=True, exist_ok=True)
    (out_root / "altered_metadata").mkdir(parents=True, exist_ok=True)

    local_stats: Dict[str, int] = {}
    local_manifest_rows: List[Dict[str, Any]] = []

    for shard_name in shard_names:
        pp = editable_v1 / "particles" / shard_name
        epth = editable_v1 / "events" / shard_name
        if not pp.is_file() or not epth.is_file():
            continue

        pdf = pd.read_parquet(pp)
        edf = pd.read_parquet(epth)
        edf_map = edf.set_index("event_id")

        p_buf: List[pd.DataFrame] = []
        e_buf: List[Dict[str, Any]] = []
        m_buf: List[Dict[str, Any]] = []

        for event_id, g in pdf.groupby("event_id", sort=True):
            eid = int(event_id)
            if eid not in id_allowed:
                continue
            g = g.sort_values("particle_index")
            md_row = edf_map.loc[eid] if eid in edf_map.index else None

            extra_kw: Dict[str, Any] = {}
            if surgery_version == "v2":
                extra_kw["seed_for_kick"] = seed

            particles_out, branch, altered_meta, delta = process_one_event_split(
                g,
                eid,
                event_in_alter_pool=True,
                md_row=md_row,
                mode=mode,
                delta_px=delta_px,
                rng=rng,
                forced_split_channel=None,
                **extra_kw,
            )
            for k, v in delta.items():
                local_stats[k] = local_stats.get(k, 0) + v

            if branch == "altered" and altered_meta is not None:
                mrow = _flatten_altered_meta_success(eid, altered_meta)
            else:
                mrow = _flatten_altered_meta_failure(
                    eid, _failure_reason_from_delta(delta)
                )

            local_stats["total_processed"] = local_stats.get("total_processed", 0) + 1

            event_row = _event_row_dict(md_row, eid, len(particles_out))
            p_buf.append(particles_out[PARTICLE_CSV_COLUMN_ORDER].copy())
            e_buf.append(event_row)
            m_buf.append(mrow)

        if not e_buf:
            continue

        si_str = shard_name.replace("shard_", "").replace(".parquet", "")
        try:
            si = int(si_str)
        except ValueError:
            si = int(e_buf[0]["event_id"]) // 10_000

        pdf_out = pd.concat(p_buf, ignore_index=True)
        edf_out = pd.DataFrame(e_buf)
        mdf_out = pd.DataFrame(m_buf)

        for col in ALTERED_META_COLUMNS:
            if col not in mdf_out.columns:
                mdf_out[col] = pd.NA
        mdf_out = mdf_out[ALTERED_META_COLUMNS]
        pdf_out = _cast_particles_df(pdf_out[PARTICLE_CSV_COLUMN_ORDER])
        edf_out = _cast_events_df(edf_out[EVENT_TABLE_COLUMNS])

        pdf_out.to_parquet(
            out_root / "particles" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        edf_out.to_parquet(
            out_root / "events" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )
        mdf_out.to_parquet(
            out_root / "altered_metadata" / f"shard_{si:06d}.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )

        fe = int(edf_out["event_id"].min())
        le = int(edf_out["event_id"].max())
        ne = int(len(edf_out))
        for dataset, nrows in (
            ("particles", int(len(pdf_out))),
            ("events", ne),
            ("altered_metadata", int(len(mdf_out))),
        ):
            local_manifest_rows.append(
                {
                    "dataset": dataset,
                    "shard_path": f"{dataset}/shard_{si:06d}.parquet",
                    "first_event_id": fe,
                    "last_event_id": le,
                    "n_events": ne,
                    "n_rows": nrows,
                }
            )

    return {"stats": local_stats, "manifest_rows": local_manifest_rows}


def _write_run_summary(
    out_parent: Path,
    stats: Dict[str, Any],
    *,
    editable_v1: Path,
    events_per_shard: int,
) -> None:
    stats["output_root"] = str(out_parent / OUTPUT_SUBDIR)
    stats["events_per_shard"] = int(events_per_shard)
    stats["output_subdir"] = OUTPUT_SUBDIR
    stats["altered_meta_columns"] = ALTERED_META_COLUMNS
    stats["editable_source"] = str(editable_v1)
    summary_path = out_parent / OUTPUT_SUBDIR / "run_summary.json"
    summary_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Transform all editable_source_v1 events through diquark+kick (no 90/10)."
    )
    ap.add_argument(
        "--editable-parent",
        type=Path,
        default=None,
        help=f"Parent of {EDITABLE_SOURCE_V1_DIRNAME}/ (default: {default_editable_parent()})",
    )
    ap.add_argument(
        "--out-parent",
        type=Path,
        default=None,
        help=f"Parent for output {OUTPUT_SUBDIR}/ (default: {default_out_parent()})",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed (reserved; split topology is fixed).")
    ap.add_argument("--delta-px", type=float, default=0.4)
    ap.add_argument("--events-per-shard", type=int, default=10_000)
    ap.add_argument("--max-events", type=int, default=0, help="If >0, only process event_id in 0..N-1.")
    ap.add_argument(
        "--target-altered-success",
        type=int,
        default=0,
        help="If >0, stop after writing this many successful alterations (skip writing failures). "
        "Scans editable shards in order until the count is reached.",
    )
    ap.add_argument(
        "--mode",
        choices=("diquark_kick", "breit_px_kick_only"),
        default="diquark_kick",
    )
    ap.add_argument(
        "--surgery-version",
        choices=("v1", "v2"),
        default="v1",
        help=(
            "v1: original fixed-fraction surgery with constant kick. "
            "v2: x_struck partition + sin(theta) kick."
        ),
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help=(
            "Parallel worker processes (spawn). Default 1 (serial). "
            "Forced to 1 when --target-altered-success > 0 or --max-events > 0."
        ),
    )
    args = ap.parse_args()

    cap = int(args.max_events)
    target_ok = int(args.target_altered_success)
    n_workers_req = max(1, int(args.workers))

    if (cap > 0 or target_ok > 0) and n_workers_req > 1:
        print(
            "transform_editable_all_altered_parquet: --max-events or --target-altered-success "
            "forces single worker (ordered scan required).",
            file=sys.stderr,
        )
        n_workers_req = 1

    parallel = cap == 0 and target_ok == 0 and n_workers_req > 1

    if args.surgery_version == "v2":
        from split_dis_sample_diquark_kick_v2 import process_one_event_split  # noqa: WPS433
    else:
        from split_dis_sample_diquark_kick import process_one_event_split  # noqa: WPS433,F811

    ep = (args.editable_parent or default_editable_parent()).resolve()
    editable_v1 = ep / EDITABLE_SOURCE_V1_DIRNAME
    if not editable_v1.is_dir():
        raise SystemExit(f"missing editable dataset: {editable_v1}")

    out_parent = (args.out_parent or default_out_parent()).resolve()
    n_accepted = _read_editable_run_summary(editable_v1)
    all_ids = list(range(n_accepted))
    if cap > 0:
        all_ids = all_ids[:cap]
    id_allowed: Set[int] = set(all_ids)

    t0 = time.perf_counter()

    if parallel:
        shard_names_sorted = sorted(
            p.name for p in (editable_v1 / "particles").glob("shard_*.parquet")
        )
        batches = [b for b in _split_list(shard_names_sorted, n_workers_req) if b]
        n_launched = len(batches)
        out_root = out_parent / OUTPUT_SUBDIR
        out_root.mkdir(parents=True, exist_ok=True)

        base_payload = {
            "editable_v1": str(editable_v1),
            "out_root": str(out_root),
            "id_allowed": list(id_allowed),
            "seed": int(args.seed),
            "delta_px": float(args.delta_px),
            "mode": str(args.mode),
            "surgery_version": str(args.surgery_version),
        }
        payloads = [{**base_payload, "shard_names": batch} for batch in batches]

        ctx = mp.get_context("spawn")
        aggregated_stats: Dict[str, Any] = {}
        aggregated_manifest: List[Dict[str, Any]] = []

        with ctx.Pool(processes=n_launched) as pool, tqdm(
            total=n_launched,
            desc="Alter shards (parallel)",
            unit="batch",
            mininterval=0.3,
        ) as pbar:
            for part in pool.imap_unordered(_alter_shard_worker, payloads):
                for k, v in part["stats"].items():
                    aggregated_stats[k] = aggregated_stats.get(k, 0) + v
                aggregated_manifest.extend(part["manifest_rows"])
                pbar.update(1)

        pd.DataFrame(aggregated_manifest).to_parquet(
            out_root / "manifest.parquet",
            index=False,
            compression="snappy",
            engine="pyarrow",
        )

        stats = dict(aggregated_stats)
        stats["mode"] = str(args.mode)
        stats["surgery_version"] = str(args.surgery_version)
        stats["seed"] = int(args.seed)
        stats["delta_px"] = float(args.delta_px)
        stats["workers"] = int(n_launched)
        stats["elapsed_s"] = float(time.perf_counter() - t0)
        _write_run_summary(out_parent, stats, editable_v1=editable_v1, events_per_shard=int(args.events_per_shard))
        return

    rng = random.Random(int(args.seed))
    writer = _AlteredAllWriter(out_parent, int(args.events_per_shard))

    stats: Dict[str, Any] = {
        "mode": args.mode,
        "surgery_version": str(args.surgery_version),
        "seed": args.seed,
        "delta_px": args.delta_px,
        "total_processed": 0,
        "written_altered": 0,
        "fallback_unchanged_no_ud_diquark": 0,
        "fallback_unchanged_struck_fail": 0,
        "fallback_unchanged_validation_fail": 0,
        "altered_channel_C": 0,
    }

    pairs = _iter_input_shard_pairs(editable_v1)
    last_report = t0
    progress_total = target_ok if target_ok > 0 else len(id_allowed)
    altered_success_written = 0
    stop = False

    extra_kw_serial: Dict[str, Any] = {}
    if args.surgery_version == "v2":
        extra_kw_serial["seed_for_kick"] = int(args.seed)

    with tqdm(total=progress_total, desc="Alter all", unit="ev", mininterval=0.3) as pbar:
        for pp, epth in pairs:
            if stop:
                break
            pdf = pd.read_parquet(pp)
            edf = pd.read_parquet(epth)
            edf_map = edf.set_index("event_id")
            for event_id, g in pdf.groupby("event_id", sort=True):
                if stop:
                    break
                eid = int(event_id)
                if eid not in id_allowed:
                    continue
                g = g.sort_values("particle_index")
                md_row = edf_map.loc[eid] if eid in edf_map.index else None
                particles_out, branch, altered_meta, delta = process_one_event_split(
                    g,
                    eid,
                    event_in_alter_pool=True,
                    md_row=md_row,
                    mode=str(args.mode),
                    delta_px=float(args.delta_px),
                    rng=rng,
                    forced_split_channel=None,
                    **extra_kw_serial,
                )
                for k, v in delta.items():
                    stats[k] = stats.get(k, 0) + v

                if branch == "altered" and altered_meta is not None:
                    mrow = _flatten_altered_meta_success(eid, altered_meta)
                else:
                    mrow = _flatten_altered_meta_failure(
                        eid, _failure_reason_from_delta(delta)
                    )

                stats["total_processed"] += 1

                if target_ok > 0:
                    if branch == "altered" and altered_meta is not None:
                        event_row = _event_row_dict(md_row, eid, len(particles_out))
                        writer.append(particles_out, event_row, mrow)
                        altered_success_written += 1
                        pbar.update(1)
                        if altered_success_written >= target_ok:
                            stop = True
                else:
                    event_row = _event_row_dict(md_row, eid, len(particles_out))
                    writer.append(particles_out, event_row, mrow)
                    pbar.update(1)

                now = time.perf_counter()
                if now - last_report >= 0.5:
                    dt = now - t0
                    rate = stats["total_processed"] / dt if dt > 0 else 0.0
                    pbar.set_postfix(rate=f"{rate:.1f} ev/s", refresh=False)
                    last_report = now

    writer.finalize()

    if target_ok > 0 and altered_success_written < target_ok:
        raise SystemExit(
            f"--target-altered-success={target_ok} but only {altered_success_written} "
            f"successful alterations were collected after scanning {stats['total_processed']} "
            f"input events (increase editable pool or relax topology)."
        )

    elapsed = time.perf_counter() - t0
    stats["elapsed_s"] = float(elapsed)
    stats["workers"] = 1
    if target_ok > 0:
        stats["target_altered_success"] = int(target_ok)
        stats["altered_success_written"] = int(altered_success_written)
    _write_run_summary(out_parent, stats, editable_v1=editable_v1, events_per_shard=int(args.events_per_shard))


if __name__ == "__main__":
    main()

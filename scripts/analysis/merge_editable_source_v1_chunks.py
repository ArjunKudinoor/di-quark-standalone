#!/usr/bin/env python3
"""
Concatenate several ``editable_source_v1`` chunk directories into one dataset.

Each chunk must have disjoint ``event_id`` ranges (e.g. parallel workers with distinct
``--event-id-offset``). The merged dataset is written as one or more shard pairs
(``particles/shard_K.parquet`` / ``events/shard_K.parquet``) using
``shard_index = event_id // events_per_shard`` (same convention as generation).

Usage::

  python merge_editable_source_v1_chunks.py \\
    --output-parent /path/to/merged \\
    --chunk-parents /path/a /path/b ...

"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANAL = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from editable_source_parquet import EDITABLE_SOURCE_V1_DIRNAME  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge editable_source_v1 chunk directories.")
    ap.add_argument("--output-parent", type=Path, required=True, help="Parent for merged editable_source_v1/")
    ap.add_argument(
        "--chunk-parents",
        type=Path,
        nargs="+",
        required=True,
        help="Each path is parent containing editable_source_v1/ (chunk output).",
    )
    ap.add_argument("--events-per-shard", type=int, default=10_000)
    args = ap.parse_args()

    out_parent = Path(args.output_parent).expanduser().resolve()
    chunks: List[Path] = [Path(p).expanduser().resolve() for p in args.chunk_parents]
    eps = max(1, int(args.events_per_shard))

    base_out = out_parent / EDITABLE_SOURCE_V1_DIRNAME
    (base_out / "particles").mkdir(parents=True, exist_ok=True)
    (base_out / "events").mkdir(parents=True, exist_ok=True)

    all_parts: List[pd.DataFrame] = []
    all_events: List[pd.DataFrame] = []
    manifest_rows: List[dict] = []

    for cp in chunks:
        v1 = cp / EDITABLE_SOURCE_V1_DIRNAME
        if not v1.is_dir():
            raise SystemExit(f"missing {v1}")
        for sub in ("particles", "events"):
            d = v1 / sub
            if not d.is_dir():
                raise SystemExit(f"missing {d}")
        for pp in sorted((v1 / "particles").glob("shard_*.parquet")):
            ep = v1 / "events" / pp.name
            if not ep.is_file():
                raise FileNotFoundError(ep)
            all_parts.append(pd.read_parquet(pp))
            all_events.append(pd.read_parquet(ep))

    if not all_parts:
        raise SystemExit("no particle shards found in chunks")

    pdf = pd.concat(all_parts, ignore_index=True)
    edf = pd.concat(all_events, ignore_index=True)
    pdf = pdf.sort_values(["event_id", "particle_index"], kind="mergesort").reset_index(drop=True)
    edf = edf.sort_values("event_id", kind="mergesort").reset_index(drop=True)

    eids = edf["event_id"].astype(int).tolist()
    if len(eids) != len(set(eids)):
        dup = [x for x in eids if eids.count(x) > 1][:5]
        raise SystemExit(f"duplicate event_id across chunks (example dup={dup})")

    n_ev = len(edf)

    eid = edf["event_id"].astype(np.int64)
    sid = (eid // int(eps)).astype(int)
    pdf = pdf.copy()
    pdf["_sid"] = (pdf["event_id"].astype(np.int64) // int(eps)).astype(int)

    manifest_rows: List[dict] = []
    for shard_idx in sorted(sid.unique().tolist()):
        sub_e = edf.loc[sid == shard_idx].copy()
        sub_p = pdf.loc[pdf["_sid"] == shard_idx].drop(columns=["_sid"], errors="ignore")
        sub_e = sub_e.sort_values("event_id", kind="mergesort").reset_index(drop=True)
        sub_p = sub_p.sort_values(["event_id", "particle_index"], kind="mergesort").reset_index(drop=True)

        sub_p.to_parquet(base_out / "particles" / f"shard_{shard_idx:06d}.parquet", index=False)
        sub_e.to_parquet(base_out / "events" / f"shard_{shard_idx:06d}.parquet", index=False)

        fe = int(sub_e["event_id"].min())
        le = int(sub_e["event_id"].max())
        ne = int(len(sub_e))
        manifest_rows.append(
            {
                "dataset": "particles",
                "shard_path": f"particles/shard_{shard_idx:06d}.parquet",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": int(len(sub_p)),
            }
        )
        manifest_rows.append(
            {
                "dataset": "events",
                "shard_path": f"events/shard_{shard_idx:06d}.parquet",
                "first_event_id": fe,
                "last_event_id": le,
                "n_events": ne,
                "n_rows": ne,
            }
        )
    v1 = out_parent / EDITABLE_SOURCE_V1_DIRNAME
    pd.DataFrame(manifest_rows).to_parquet(v1 / "manifest.parquet", index=False)

    rs = {
        "n_accepted": n_ev,
        "events_per_shard": eps,
        "n_output_shards": int(len(sid.unique())),
        "merge_sources": [str(c) for c in chunks],
        "merged_editable_parent": str(out_parent),
    }
    (v1 / "run_summary.json").write_text(json.dumps(rs, indent=2) + "\n", encoding="utf-8")

    print(f"[merge_editable_chunks] wrote {base_out} with {n_ev} events, {len(pdf)} particle rows.")


if __name__ == "__main__":
    main()

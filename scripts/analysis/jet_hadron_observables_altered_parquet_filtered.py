#!/usr/bin/env python3
"""
Jet–hadron observables (same physics as ``jet_hadron_observables_split_pi_pm``) for a
**subset** of altered Parquet events selected by ``split_channel`` in ``altered_metadata``.

Typical use: **signal** ``altered_100k_parquet`` uses hard channel **C**
``(ud)→[d]+(us)+sbar``.  This driver keeps events with ``alteration_succeeded == 1`` and the
requested ``split_channel`` (default **C**), reinjects from the parton table, and writes one CSV
row per altered event (leading charged π + cuts).  Optional ``--editable-parent`` adds paired
``sample=unchanged`` rows from ``editable_source_v1`` for the same ``event_id`` set (for π±
comparison plots).

Example::

  python scripts/analysis/jet_hadron_observables_altered_parquet_filtered.py \\
    --altered-root ~/Data/dis_isr_editable_altered_100k/altered_100k_parquet \\
    --split-channel C \\
    --editable-parent ~/Data/dis_isr_editable_source_100k \\
    --workers 6 \\
    --out-csv ~/Data/.../jet_hadron_observables_signal_channel_C.csv

For 5×41 GeV edits with strict ``x > 0.05`` and ``2 < Q < 4`` GeV, match generation with::

  python scripts/analysis/generate_dis_editable_source_parquet.py \\
    --kinematics-preset e5_ep41_x005_q2to4 ...

and run this driver with ``--kinematics-preset e5_ep41_x005_q2to4`` so PYTHIA beam cards and DIS
selection match the parquet pool.

For each event, the matching row in ``events/shard_*.parquet`` must include the flipped-lab beam
four-vectors (``ein_fl_*``, ``esc_fl_*``, ``pin_fl_*`` from a current
``generate_dis_editable_source_parquet`` run) so lab→Breit ``LT`` can be rebuilt for η.
Charged π candidates use **proton-forward** lab η matching paper Eq. 2 (**open** interval,
default ``2.0 < η_h < 3.5``). After ``LT^{-1}``,
momenta are mapped back through ``flip_z`` (undo ``FLIP_Z_PTREL``) before computing η.

Parallelism: ``--workers N`` (``spawn``) partitions **particle shards** across processes (one
reused ``Pythia`` per process). Use ``--max-events`` only with ``--workers 1`` (cap forces serial).

Requires: pythia8, pandas, numpy, tqdm.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

try:
    from tqdm import tqdm
except ImportError as exc:  # pragma: no cover
    raise SystemExit("tqdm is required (see requirements.txt)") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ANAL = Path(__file__).resolve().parent
for _p in (_PROJECT_ROOT, _ANAL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from editable_source_parquet import EDITABLE_SOURCE_V1_DIRNAME  # noqa: E402

from jet_hadron_observables_split_pi_pm import (  # noqa: E402
    build_pythia_reinjector,
    process_event_dataframe,
)

# Proton-forward lab η (via collider_forward_pion_pseudorapidity): strict η_min < η < η_max (open).
# Match paper Eq. 2 acceptance (2.0, 3.5) target / remnant selections.
_PION_COLLIDER_ETA_ACCEPTANCE_MIN_MAX = (2.0, 3.5)


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


def _ipc_worker_altered_shards(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Child process: one ``Pythia``, a subset of particles ``shard_*.parquet`` names."""
    altered_root = Path(bundle["altered_root"])
    allowed: Set[int] = set(bundle["allowed_eids"])
    struck_map: Dict[int, int] = bundle["struck_map"]
    ch_map: Dict[int, str] = bundle["ch_map"]
    descr_map: Dict[int, str] = bundle["descr_map"]
    want_label = str(bundle["want_label"])
    sample_label = str(bundle["sample_label"])
    proc_phys_kw = bundle["proc_phys_kw"]
    p_e = float(bundle["p_e"])
    p_p = float(bundle["p_p"])
    shard_names: List[str] = list(bundle["shard_names"])
    raw_eta = bundle.get("pion_collider_eta_acceptance")
    eta_acc: Optional[Tuple[float, float]] = (
        None if raw_eta is None else tuple(float(x) for x in raw_eta)
    )

    isr_on = bool(bundle.get("reinject_parton_isr", True))
    fsr_on = bool(bundle.get("reinject_parton_fsr", True))
    pythia = build_pythia_reinjector(
        e_beam_gev=p_e, p_beam_gev=p_p, parton_level_isr=isr_on, parton_level_fsr=fsr_on
    )
    particles_dir = altered_root / "particles"
    events_dir = altered_root / "events"
    out_rows: List[Dict[str, Any]] = []

    for name in shard_names:
        pshard = particles_dir / name
        eshard = events_dir / name
        if not eshard.is_file():
            raise FileNotFoundError(
                f"missing matching events shard {eshard.name!r} next to particles/{name!r}"
            )
        evdf_indexed = pd.read_parquet(eshard).set_index("event_id")
        pdf = pd.read_parquet(pshard)
        for eid, g in pdf.groupby("event_id", sort=True):
            eid_i = int(eid)
            if eid_i not in allowed:
                continue
            struck = struck_map.get(eid_i, -1)
            if struck < 0:
                continue
            g = g.sort_values("particle_index").reset_index(drop=True)
            if eid_i not in evdf_indexed.index:
                raise KeyError(
                    f"particles shard references event_id={eid_i} but events shard has no row"
                )
            evrow = evdf_indexed.loc[eid_i]
            if isinstance(evrow, pd.DataFrame):
                evrow = evrow.iloc[0]
            rows = process_event_dataframe(
                g,
                struck,
                sample=sample_label,
                csv_label=str(name),
                csv_momenta_breit=True,
                pythia_reuse=pythia,
                event_row=evrow,
                pion_collider_eta_acceptance=eta_acc,
                pion_breit_hemisphere="target",
                **proc_phys_kw,
            )
            if len(rows) != 1:
                raise RuntimeError(f"event {eid_i}: expected 1 observable row, got {len(rows)}")
            row = dict(rows[0])
            row["hard_split_channel"] = ch_map.get(eid_i, want_label)
            row["hard_channel_description"] = descr_map.get(eid_i, "")
            out_rows.append(row)
    return out_rows


def _ipc_worker_editable_shards(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Child process: unchanged twin for ``editable_source_v1`` shards."""
    ep = Path(bundle["editable_parent"])
    altered_eids_done: Set[int] = set(bundle["altered_eids_done"])
    struck_map: Dict[int, int] = bundle["struck_map"]
    ch_map: Dict[int, str] = bundle["ch_map"]
    descr_map: Dict[int, str] = bundle["descr_map"]
    want_label = str(bundle["want_label"])
    proc_phys_kw = bundle["proc_phys_kw"]
    p_e = float(bundle["p_e"])
    p_p = float(bundle["p_p"])
    shard_names: List[str] = list(bundle["shard_names"])
    raw_eta = bundle.get("pion_collider_eta_acceptance")
    eta_acc: Optional[Tuple[float, float]] = (
        None if raw_eta is None else tuple(float(x) for x in raw_eta)
    )

    isr_on = bool(bundle.get("reinject_parton_isr", True))
    fsr_on = bool(bundle.get("reinject_parton_fsr", True))
    pythia = build_pythia_reinjector(
        e_beam_gev=p_e, p_beam_gev=p_p, parton_level_isr=isr_on, parton_level_fsr=fsr_on
    )
    part_dir = ep / EDITABLE_SOURCE_V1_DIRNAME / "particles"
    eve_dir = ep / EDITABLE_SOURCE_V1_DIRNAME / "events"
    out_rows: List[Dict[str, Any]] = []

    for name in shard_names:
        pshard = part_dir / name
        eshard = eve_dir / name
        if not eshard.is_file():
            raise FileNotFoundError(f"missing editable events shard matching {pshard}")
        evdf_indexed = pd.read_parquet(eshard).set_index("event_id")
        pdf = pd.read_parquet(pshard)
        for eid, g in pdf.groupby("event_id", sort=True):
            eid_i = int(eid)
            if eid_i not in altered_eids_done:
                continue
            struck = struck_map.get(eid_i, -1)
            if struck < 0:
                continue
            g = g.sort_values("particle_index").reset_index(drop=True)
            if eid_i not in evdf_indexed.index:
                raise KeyError(f"editable: event_id={eid_i} missing from {eshard}")
            evrow = evdf_indexed.loc[eid_i]
            if isinstance(evrow, pd.DataFrame):
                evrow = evrow.iloc[0]
            rows = process_event_dataframe(
                g,
                struck,
                sample="unchanged",
                csv_label=f"editable:{name}",
                csv_momenta_breit=True,
                pythia_reuse=pythia,
                event_row=evrow,
                pion_collider_eta_acceptance=eta_acc,
                pion_breit_hemisphere="target",
                **proc_phys_kw,
            )
            if len(rows) != 1:
                raise RuntimeError(
                    f"unchanged event {eid_i}: expected 1 observable row, got {len(rows)}"
                )
            row = dict(rows[0])
            row["hard_split_channel"] = ch_map.get(eid_i, want_label)
            row["hard_channel_description"] = descr_map.get(eid_i, "")
            out_rows.append(row)
    return out_rows


def _load_filter_map(
    altered_root: Path, want_channel: Optional[str]
) -> Tuple[Dict[int, int], Dict[int, str], Dict[int, str]]:
    """struck_index, channel_description, and split_channel letter per event_id."""
    mdir = altered_root / "altered_metadata"
    if not mdir.is_dir():
        raise FileNotFoundError(f"missing altered_metadata under {altered_root}")
    struck: Dict[int, int] = {}
    descr: Dict[int, str] = {}
    ch_letter: Dict[int, str] = {}
    cols = ["event_id", "alteration_succeeded", "split_channel", "struck_quark_index"]
    has_desc = False
    probe = next(iter(mdir.glob("shard_*.parquet")), None)
    if probe is not None:
        pc = set(pd.read_parquet(probe, engine="pyarrow").columns)
        if "channel_description" in pc:
            has_desc = True
            cols.append("channel_description")

    for pq in sorted(mdir.glob("shard_*.parquet")):
        df = pd.read_parquet(pq, columns=cols)
        ok = df["alteration_succeeded"] == 1
        if want_channel is None:
            sub = df[ok]
        else:
            sub = df[ok & (df["split_channel"].astype(str) == want_channel)]
        for _, r in sub.iterrows():
            eid = int(r["event_id"])
            struck[eid] = int(r["struck_quark_index"])
            ch_letter[eid] = str(r.get("split_channel", ""))
            if has_desc:
                descr[eid] = str(r.get("channel_description", ""))
    return struck, descr, ch_letter


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Jet–hadron observables for altered Parquet events filtered by split_channel."
    )
    ap.add_argument(
        "--altered-root",
        type=Path,
        required=True,
        help="Directory with particles/, events/, altered_metadata/ (e.g. altered_100k_parquet).",
    )
    ap.add_argument(
        "--split-channel",
        choices=("C",),
        default="C",
        help='Hard topology filter on metadata split_channel (default "C" = (ud)→[d]+(us)+sbar). Ignored if --all-channels.',
    )
    ap.add_argument(
        "--all-channels",
        action="store_true",
        help="Include every alteration_succeeded event (any split_channel).",
    )
    ap.add_argument(
        "--editable-parent",
        type=Path,
        default=None,
        help=(
            "Parent directory containing editable_source_v1/ (same pool as altered). "
            "If set, append unchanged reinjection rows for the same filtered event_ids."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output observables CSV.",
    )
    ap.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="If >0, stop after this many matching events (debug).",
    )
    ap.add_argument(
        "--sample-label",
        type=str,
        default="altered",
        help='Value written in the ``sample`` column (default "altered").',
    )
    ap.add_argument(
        "--kinematics-preset",
        choices=("legacy", "e5_ep41_x005_q2to4"),
        default="legacy",
        help=(
            "'e5_ep41_x005_q2to4': nominal PYTHIA beams 5×41 GeV, strict x > 0.05 and 2 < Q < 4 GeV "
            "(match editable generation with ``--kinematics-preset e5_ep41_x005_q2to4``). "
            "'legacy' uses 18×275 GeV and global x,Q windows from ``analyze_events_raw``."
        ),
    )
    ap.add_argument(
        "--beam-e",
        type=float,
        default=None,
        help="PYTHIA Beams:eA in GeV (default from --kinematics-preset).",
    )
    ap.add_argument(
        "--beam-p",
        type=float,
        default=None,
        help="PYTHIA Beams:eB (proton) in GeV (default from --kinematics-preset).",
    )
    ap.add_argument(
        "--dis-x-floor",
        type=float,
        default=None,
        help=(
            "Require Bjorken x strictly above this (overrides preset default when preset sets x; "
            "with preset 'legacy', optional floor instead of xmin_ptrel)."
        ),
    )
    ap.add_argument(
        "--dis-q-min",
        type=float,
        default=None,
        help="Together with --dis-q-max: strict Q window q_min < Q < q_max (GeV).",
    )
    ap.add_argument(
        "--dis-q-max",
        type=float,
        default=None,
        help="See --dis-q-min.",
    )
    ap.add_argument("--pion-pt-min", type=float, default=None)
    ap.add_argument("--pion-pt-max", type=float, default=None)
    ap.add_argument("--jet-pt-min", type=float, default=None)
    ap.add_argument("--jet-pt-max", type=float, default=None)
    ap.add_argument(
        "--reinject-parton-isr",
        choices=("on", "off"),
        default="on",
        help="PYTHIA PartonLevel:ISR during reinjection (default on).",
    )
    ap.add_argument(
        "--reinject-parton-fsr",
        choices=("on", "off"),
        default="on",
        help="PYTHIA PartonLevel:FSR during reinjection (default on).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel processes (spawn); shards are split across workers. Ignored if --max-events > 0.",
    )
    ap.add_argument(
        "--no-pion-eta-filter",
        action="store_true",
        help="Disable charged-π lab pseudorapidity filtering before leading-pion selection.",
    )
    ap.add_argument(
        "--pion-eta-min",
        type=float,
        default=_PION_COLLIDER_ETA_ACCEPTANCE_MIN_MAX[0],
        help="Charged π lab-frame η lower bound (ignored with --no-pion-eta-filter).",
    )
    ap.add_argument(
        "--pion-eta-max",
        type=float,
        default=_PION_COLLIDER_ETA_ACCEPTANCE_MIN_MAX[1],
        help="Charged π lab-frame η upper bound (ignored with --no-pion-eta-filter).",
    )
    args = ap.parse_args()

    if args.dis_q_min is not None or args.dis_q_max is not None:
        if args.dis_q_min is None or args.dis_q_max is None:
            raise SystemExit("--dis-q-min and --dis-q-max must both be set, or omit both.")
    if args.pion_pt_min is not None or args.pion_pt_max is not None:
        if args.pion_pt_min is None or args.pion_pt_max is None:
            raise SystemExit("--pion-pt-min and --pion-pt-max must both be set, or omit both.")
        if float(args.pion_pt_min) >= float(args.pion_pt_max):
            raise SystemExit("--pion-pt-min must be < --pion-pt-max")
    if args.jet_pt_min is not None or args.jet_pt_max is not None:
        if args.jet_pt_min is None or args.jet_pt_max is None:
            raise SystemExit("--jet-pt-min and --jet-pt-max must both be set, or omit both.")
        if float(args.jet_pt_min) >= float(args.jet_pt_max):
            raise SystemExit("--jet-pt-min must be < --jet-pt-max")

    preset = str(args.kinematics_preset)
    q_open_iv: Optional[Tuple[float, float]]
    x_exc: Optional[float]
    if preset == "e5_ep41_x005_q2to4":
        p_e = float(args.beam_e) if args.beam_e is not None else 5.0
        p_p = float(args.beam_p) if args.beam_p is not None else 41.0
        x_exc = float(args.dis_x_floor) if args.dis_x_floor is not None else 0.05
        if args.dis_q_min is None:
            q_open_iv = (2.0, 4.0)
        else:
            q_open_iv = (float(args.dis_q_min), float(args.dis_q_max))
    else:
        p_e = float(args.beam_e) if args.beam_e is not None else 18.0
        p_p = float(args.beam_p) if args.beam_p is not None else 275.0
        x_exc = float(args.dis_x_floor) if args.dis_x_floor is not None else None
        q_open_iv = (
            None
            if args.dis_q_min is None
            else (float(args.dis_q_min), float(args.dis_q_max))
        )

    altered_root = args.altered_root.expanduser().resolve()
    want_filter: Optional[str] = None if bool(args.all_channels) else str(args.split_channel)
    want_label = "all" if want_filter is None else want_filter
    struck_map, desc_map, ch_map = _load_filter_map(altered_root, want_filter)
    allowed: Set[int] = set(struck_map.keys())
    if not allowed:
        raise SystemExit(
            f"no matching altered events under {altered_root} (filter={want_label!r})"
        )

    proc_phys_kw = dict(
        pythia_e_beam_gev=p_e,
        pythia_p_beam_gev=p_p,
        dis_x_exclusive_floor=x_exc,
        dis_q_open_interval_gev=q_open_iv,
        pion_pt_open_interval_gev=(
            None
            if args.pion_pt_min is None
            else (float(args.pion_pt_min), float(args.pion_pt_max))
        ),
        jet_pt_open_interval_gev=(
            None
            if args.jet_pt_min is None
            else (float(args.jet_pt_min), float(args.jet_pt_max))
        ),
        pythia_parton_level_isr=(str(args.reinject_parton_isr) == "on"),
        pythia_parton_level_fsr=(str(args.reinject_parton_fsr) == "on"),
    )
    reinject_isr_on = proc_phys_kw["pythia_parton_level_isr"]
    reinject_fsr_on = proc_phys_kw["pythia_parton_level_fsr"]
    if bool(args.no_pion_eta_filter):
        eta_acc: Optional[Tuple[float, float]] = None
    else:
        eta_acc = (float(args.pion_eta_min), float(args.pion_eta_max))
        if eta_acc[0] >= eta_acc[1]:
            raise SystemExit("--pion-eta-min must be < --pion-eta-max")

    particles_dir = altered_root / "particles"
    events_dir = altered_root / "events"
    if not particles_dir.is_dir():
        raise FileNotFoundError(f"missing particles under {altered_root}")

    cap = int(args.max_events)
    workers_requested = max(1, int(args.workers))
    n_workers_req = workers_requested
    if cap > 0 and n_workers_req > 1:
        print(
            "jet_hadron_observables_altered_parquet_filtered: --max-events > 0 forces single worker.",
            file=sys.stderr,
        )
        n_workers_req = 1
    parallel = cap <= 0 and n_workers_req > 1

    all_rows: List[Dict[str, Any]] = []
    shard_names_sorted = sorted(p.name for p in particles_dir.glob("shard_*.parquet"))
    n_workers_eff_altered = 1
    n_workers_eff_editable = 1
    pythia_serial: Any = None  # reused for altered + unchanged in serial mode

    if parallel:
        batches = _split_list(list(shard_names_sorted), n_workers_req)
        batches = [b for b in batches if b]
        n_launched = len(batches)
        n_workers_eff_altered = n_launched
        base_bundle = dict(
            altered_root=str(altered_root),
            allowed_eids=list(allowed),
            struck_map=struck_map,
            ch_map=ch_map,
            descr_map=desc_map,
            want_label=want_label,
            sample_label=str(args.sample_label),
            proc_phys_kw=proc_phys_kw,
            p_e=p_e,
            p_p=p_p,
            pion_collider_eta_acceptance=eta_acc,
            reinject_parton_isr=reinject_isr_on,
            reinject_parton_fsr=reinject_fsr_on,
        )
        ctx = mp.get_context("spawn")
        payloads = [{**base_bundle, "shard_names": batch} for batch in batches]
        with ctx.Pool(processes=n_launched) as pool, tqdm(
            total=n_launched, unit="shard-batch", desc=f"altered×{want_label}", mininterval=0.3
        ) as pbar:
            for part in pool.imap_unordered(_ipc_worker_altered_shards, payloads):
                all_rows.extend(part)
                pbar.update(1)
    else:
        pythia_serial = build_pythia_reinjector(
            e_beam_gev=p_e,
            p_beam_gev=p_p,
            parton_level_isr=reinject_isr_on,
            parton_level_fsr=reinject_fsr_on,
        )
        total_bar = len(allowed) if cap <= 0 else min(len(allowed), cap)
        stop = False
        n_seen = 0
        with tqdm(total=total_bar, unit="ev", desc=f"channel {want_label}", mininterval=0.2) as pbar:
            for pshard in sorted(particles_dir.glob("shard_*.parquet")):
                if stop:
                    break
                eshard = events_dir / pshard.name
                if not eshard.is_file():
                    raise FileNotFoundError(
                        f"missing matching events shard {eshard.name!r} next to particles/{pshard.name!r}; "
                        "need events table with flipped-lab beam columns for π η acceptance."
                    )
                evdf_indexed = pd.read_parquet(eshard).set_index("event_id")
                pdf = pd.read_parquet(pshard)
                for eid, g in pdf.groupby("event_id", sort=True):
                    eid_i = int(eid)
                    if eid_i not in allowed:
                        continue
                    if cap > 0 and n_seen >= cap:
                        stop = True
                        break
                    struck = struck_map.get(eid_i, -1)
                    if struck < 0:
                        continue
                    g = g.sort_values("particle_index").reset_index(drop=True)
                    if eid_i not in evdf_indexed.index:
                        raise KeyError(
                            f"particles shard references event_id={eid_i} but events shard has no row"
                        )
                    evrow = evdf_indexed.loc[eid_i]
                    if isinstance(evrow, pd.DataFrame):
                        evrow = evrow.iloc[0]
                    rows = process_event_dataframe(
                        g,
                        struck,
                        sample=str(args.sample_label),
                        csv_label=str(pshard.name),
                        csv_momenta_breit=True,
                        pythia_reuse=pythia_serial,
                        event_row=evrow,
                        pion_collider_eta_acceptance=eta_acc,
                        pion_breit_hemisphere="target",
                        **proc_phys_kw,
                    )
                    if len(rows) != 1:
                        raise RuntimeError(
                            f"event {eid_i}: expected 1 observable row, got {len(rows)}"
                        )
                    row = dict(rows[0])
                    row["hard_split_channel"] = ch_map.get(eid_i, want_label)
                    row["hard_channel_description"] = desc_map.get(eid_i, "")
                    all_rows.append(row)
                    n_seen += 1
                    pbar.update(1)

    altered_eids_done = {
        int(r["event_id"])
        for r in all_rows
        if str(r.get("sample", "")) == str(args.sample_label)
    }

    if args.editable_parent is not None:
        ep = args.editable_parent.expanduser().resolve()
        part_dir = ep / EDITABLE_SOURCE_V1_DIRNAME / "particles"
        eve_dir = ep / EDITABLE_SOURCE_V1_DIRNAME / "events"
        if not part_dir.is_dir():
            raise FileNotFoundError(f"missing editable particles dir: {part_dir}")
        if not eve_dir.is_dir():
            raise FileNotFoundError(f"missing editable events dir: {eve_dir}")
        editable_shard_names = sorted(p.name for p in part_dir.glob("shard_*.parquet"))

        if parallel:
            eb_batches = _split_list(list(editable_shard_names), n_workers_req)
            eb_batches = [b for b in eb_batches if b]
            ed_base = dict(
                editable_parent=str(ep.resolve()),
                altered_eids_done=list(altered_eids_done),
                struck_map=struck_map,
                ch_map=ch_map,
                descr_map=desc_map,
                want_label=want_label,
                proc_phys_kw=proc_phys_kw,
                p_e=p_e,
                p_p=p_p,
                pion_collider_eta_acceptance=eta_acc,
                reinject_parton_isr=reinject_isr_on,
                reinject_parton_fsr=reinject_fsr_on,
            )
            ctx2 = mp.get_context("spawn")
            ed_payloads = [{**ed_base, "shard_names": b} for b in eb_batches]
            n_workers_eff_editable = len(ed_payloads)
            n_extra = 0
            with ctx2.Pool(processes=len(ed_payloads)) as pool2, tqdm(
                total=len(ed_payloads),
                unit="shard-batch",
                desc="unchanged (editable)×par",
                mininterval=0.3,
            ) as pbar_e:
                for part in pool2.imap_unordered(_ipc_worker_editable_shards, ed_payloads):
                    all_rows.extend(part)
                    n_extra += len(part)
                    pbar_e.update(1)
            if n_extra != len(altered_eids_done):
                raise RuntimeError(
                    f"unchanged pairing: expected {len(altered_eids_done)} rows, wrote {n_extra}"
                )
        else:
            assert pythia_serial is not None
            n_un = 0
            with tqdm(
                total=len(altered_eids_done),
                unit="ev",
                desc="unchanged (editable)",
                mininterval=0.2,
            ) as pbar_u:
                for pshard in sorted(part_dir.glob("shard_*.parquet")):
                    eshard = eve_dir / pshard.name
                    if not eshard.is_file():
                        raise FileNotFoundError(
                            f"missing editable events shard matching {pshard}"
                        )
                    evdf_indexed = pd.read_parquet(eshard).set_index("event_id")
                    pdf = pd.read_parquet(pshard)
                    for eid, g in pdf.groupby("event_id", sort=True):
                        eid_i = int(eid)
                        if eid_i not in altered_eids_done:
                            continue
                        struck = struck_map.get(eid_i, -1)
                        if struck < 0:
                            continue
                        g = g.sort_values("particle_index").reset_index(drop=True)
                        if eid_i not in evdf_indexed.index:
                            raise KeyError(f"editable: event_id={eid_i} missing from {eshard}")
                        evrow = evdf_indexed.loc[eid_i]
                        if isinstance(evrow, pd.DataFrame):
                            evrow = evrow.iloc[0]
                        rows = process_event_dataframe(
                            g,
                            struck,
                            sample="unchanged",
                            csv_label=f"editable:{pshard.name}",
                            csv_momenta_breit=True,
                            pythia_reuse=pythia_serial,
                            event_row=evrow,
                            pion_collider_eta_acceptance=eta_acc,
                            pion_breit_hemisphere="target",
                            **proc_phys_kw,
                        )
                        if len(rows) != 1:
                            raise RuntimeError(
                                f"unchanged event {eid_i}: expected 1 observable row, got {len(rows)}"
                            )
                        row = dict(rows[0])
                        row["hard_split_channel"] = ch_map.get(eid_i, want_label)
                        row["hard_channel_description"] = desc_map.get(eid_i, "")
                        all_rows.append(row)
                        n_un += 1
                        pbar_u.update(1)
            if n_un != len(altered_eids_done):
                raise RuntimeError(
                    f"unchanged pairing: expected {len(altered_eids_done)} rows, wrote {n_un}"
                )

    all_rows.sort(
        key=lambda r: (str(r.get("sample", "")), int(r.get("event_id", -1)))
    )

    out = Path(args.out_csv).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)
    n_ok = sum(1 for r in all_rows if r.get("ok"))
    print(
        json.dumps(
            {
                "altered_root": str(altered_root),
                "split_channel": want_label,
                "n_rows_written": len(all_rows),
                "n_altered_events": len(altered_eids_done),
                "n_ok": int(n_ok),
                "editable_parent": str(args.editable_parent) if args.editable_parent else "",
                "out_csv": str(out),
                "kinematics_preset": preset,
                "pythia_e_beam_gev": float(p_e),
                "pythia_p_beam_gev": float(p_p),
                "dis_x_exclusive_floor": float(x_exc) if x_exc is not None else None,
                "dis_Q_open_gev": [float(q_open_iv[0]), float(q_open_iv[1])]
                if q_open_iv is not None
                else None,
                "workers_requested": int(workers_requested),
                "workers_effective_parallel": bool(parallel),
                "shard_worker_processes_altered": int(n_workers_eff_altered),
                "shard_worker_processes_editable": int(n_workers_eff_editable),
                "pion_collider_eta_acceptance": None if eta_acc is None else list(eta_acc),
                "reinject_parton_isr": bool(reinject_isr_on),
                "reinject_parton_fsr": bool(reinject_fsr_on),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

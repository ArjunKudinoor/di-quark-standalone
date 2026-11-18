#!/usr/bin/env python3
"""Channel-C Track A rehadronization, with paired leading-proton/neutron rows.

Same contiguous shard batches, sorted event order, separate altered/unchanged
worker pools and original PYTHIA RNG as the pi0 driver. Workers stream temporary
CSV batches outside the repo instead of retaining paper-scale rows in RAM.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import tempfile
import pandas as pd
from tqdm import tqdm

from jet_hadron_observables_altered_parquet_filtered import _load_filter_map, _split_list
from jet_hadron_observables_split_baryons import build_pythia_reinjector, process_event_dataframe, ZLC_MAX
from validate_fig3_pi0_inputs import validate as validate_track_a_inputs

FIELDS = [
    'event_id', 'sample', 'csv', 'baryon', 'pdg_id', 'ok', 'reason', 'Q2', 'xB',
    'color_ok', 'pythia_ok', 'n_final_hadrons', 'baryon_lab_eta',
    'baryon_breit_energy_GeV', 'baryon_breit_pz_GeV', 'baryon_pht_GeV', 'jet_pjt_GeV',
    'zLC', 'phi_hJ', 'delta_P_hJ_GeV', 'Pbar_t_GeV', 'sum_mag_GeV',
    'pythia_error', 'color_message', 'hard_split_channel', 'hard_channel_description',
]


def _process_shards(bundle, sample):
    root = Path(bundle['root'])
    path = Path(bundle['part_csv'])
    struck = bundle['struck_map']
    phys = bundle['physics']
    pythia = bundle.get('pythia_reuse')
    if pythia is None:
        pythia = build_pythia_reinjector(e_beam_gev=phys['pythia_e_beam_gev'],
                                        p_beam_gev=phys['pythia_p_beam_gev'])
    counts, reasons, written, previous = Counter(), Counter(), 0, None
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for name in bundle['shard_names']:
            events = pd.read_parquet(root/'events'/name).set_index('event_id')
            if not events.index.is_unique:
                raise ValueError(f'Duplicate event IDs in {root}/events/{name}')
            particles = pd.read_parquet(root/'particles'/name)
            for eid, group in particles.groupby('event_id', sort=True):
                eid = int(eid)
                if eid not in struck:
                    continue
                if previous is not None and eid <= previous:
                    raise ValueError('Source IDs are not increasing across the sorted shard batch')
                previous = eid
                if eid not in events.index:
                    raise KeyError(f'Missing event {eid} in {name}')
                rows = process_event_dataframe(
                    group, struck[eid], sample=sample,
                    csv_label=f'editable:{name}' if sample == 'unchanged' else name,
                    csv_momenta_breit=True, pythia_reuse=pythia, event_row=events.loc[eid],
                    **phys,
                )
                if len(rows) != 1:
                    raise RuntimeError(f'Event {eid}: expected exactly one row')
                row = dict(rows[0])
                row.update(hard_split_channel='C', hard_channel_description='(ud)->[d]+(us)+sbar')
                if row['ok']:
                    eta_min, eta_max = phys['baryon_collider_eta_acceptance']
                    q_min, q_max = phys['dis_q_open_interval_gev']
                    if not (eta_min < row['baryon_lab_eta'] < eta_max
                            and row['baryon_breit_pz_GeV'] > 0
                            and phys['dis_x_exclusive_floor'] < row['xB']
                            and q_min*q_min < row['Q2'] < q_max*q_max
                            and .1 <= row['zLC'] <= ZLC_MAX
                            and 0 <= row['phi_hJ'] <= 3.141592653589793
                            and row['pdg_id'] in (2212, 2112)):
                        raise RuntimeError(f'Accepted baryon failed validation: {eid}')
                    counts[row['baryon']] += 1
                reasons[row['reason']] += 1
                writer.writerow(row)
                written += 1
    return dict(sample=sample, path=str(path), rows=written, accepted=dict(counts), reasons=dict(reasons))


def _ipc_worker_altered_shards(bundle):
    return _process_shards(bundle, 'altered')


def _ipc_worker_editable_shards(bundle):
    return _process_shards(bundle, 'unchanged')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--altered-root', type=Path, required=True)
    ap.add_argument('--editable-parent', type=Path, required=True)
    ap.add_argument('--out-csv', type=Path, required=True)
    ap.add_argument('--summary-json', type=Path, required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--max-events', type=int, default=0,
                    help='Optional matching source-ID cap; forces serial, as in the original driver.')
    ap.add_argument('--beam-e', type=float, default=5.)
    ap.add_argument('--beam-p', type=float, default=41.)
    ap.add_argument('--dis-x-floor', type=float, default=.05)
    ap.add_argument('--dis-q-min', type=float, default=4.)
    ap.add_argument('--dis-q-max', type=float, default=5.)
    ap.add_argument('--baryon-eta-min', type=float, default=2.)
    ap.add_argument('--baryon-eta-max', type=float, default=3.5)
    args = ap.parse_args()
    if args.workers < 1 or args.max_events < 0:
        ap.error('workers must be positive and max-events nonnegative')
    if not (args.dis_q_min < args.dis_q_max and args.baryon_eta_min < args.baryon_eta_max):
        ap.error('Invalid Q or eta interval')
    parent = args.editable_parent.expanduser().resolve()
    altered = args.altered_root.expanduser().resolve()
    out = args.out_csv.expanduser().resolve()
    summary_path = args.summary_json.expanduser().resolve()
    repo = Path(__file__).resolve().parents[2]
    data_base = (Path.home()/'Data/dis_q_bins_e5_ep41_standalone').resolve()
    if not out.is_relative_to(data_base) or not any(p.endswith('_baryons') for p in out.parts):
        ap.error('Large CSV must be inside a separate *_baryons directory under the Track A data root')
    if not summary_path.is_relative_to(repo/'outputs'):
        ap.error('Small summary must be under repository outputs/')
    if out.exists() or summary_path.exists():
        raise SystemExit('Refusing to overwrite an existing baryon CSV or summary')
    source_summary = json.loads((altered/'run_summary.json').read_text())
    expected = int(source_summary['total_processed'])
    report = validate_track_a_inputs(parent, expected)
    if not report['reuse_A1_A2']:
        raise ValueError('Missing editable source')
    if (source_summary.get('surgery_version') != 'v2' or source_summary.get('seed') != 42
            or source_summary.get('delta_px') != .4 or not 0 < expected <= 9600000):
        raise ValueError('Require original v2/seed-42/0.4-GeV source settings, at most 9.6M events')
    struck, descriptions, channels = _load_filter_map(altered, 'C')
    del descriptions, channels
    struck = {eid: idx for eid, idx in struck.items() if idx >= 0}
    if args.max_events:
        struck = {eid: struck[eid] for eid in sorted(struck)[:args.max_events]}
    if not struck:
        raise ValueError('No successful channel-C events with a resolved struck index')
    workers = 1 if args.max_events else args.workers
    physics = dict(
        pythia_e_beam_gev=args.beam_e, pythia_p_beam_gev=args.beam_p,
        dis_x_exclusive_floor=args.dis_x_floor,
        dis_q_open_interval_gev=(args.dis_q_min, args.dis_q_max),
        baryon_collider_eta_acceptance=(args.baryon_eta_min, args.baryon_eta_max),
        baryon_breit_hemisphere='target',
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    # Original serial mode reuses one RNG stream across both arms.
    serial_pythia = (build_pythia_reinjector(e_beam_gev=args.beam_e, p_beam_gev=args.beam_p)
                     if workers == 1 else None)
    # Only this newly-created temporary directory is removed on completion.
    # Original source and any pre-existing baryon data are never deleted.
    with tempfile.TemporaryDirectory(prefix=out.stem+'_work_', dir=out.parent) as tmp:
        for sample, root, worker in [
            ('altered', altered, _ipc_worker_altered_shards),
            ('unchanged', parent/'editable_source_v1', _ipc_worker_editable_shards),
        ]:
            names = sorted(p.name for p in (root/'particles').glob('shard_*.parquet'))
            if not names:
                raise ValueError(f'No particle shards under {root}')
            payloads = [dict(root=str(root), shard_names=batch, struck_map=struck, physics=physics,
                             part_csv=str(Path(tmp)/f'{sample}_{i:04d}.csv'))
                        for i, batch in enumerate(_split_list(names, workers))]
            if workers == 1:
                payloads[0]['pythia_reuse'] = serial_pythia
                arm = [worker(payloads[0])]
            else:
                with mp.get_context('spawn').Pool(len(payloads)) as pool:
                    arm = list(tqdm(pool.imap(worker, payloads), total=len(payloads),
                                    desc=f'{sample} baryons', unit='shard-batch'))
            if sum(r['rows'] for r in arm) != len(struck):
                raise RuntimeError(f'{sample}: expected exactly {len(struck)} paired source rows')
            results.extend(arm)
        # Each arm's batches are already ordered by shard and then event_id.
        staged = Path(tmp)/'complete.csv'
        header = None
        with staged.open('x') as dest:
            for result in results:
                with Path(result['path']).open() as source:
                    current_header = source.readline()
                    if header is None:
                        header = current_header
                        dest.write(header)
                    elif current_header != header:
                        raise RuntimeError('Worker CSV schemas differ')
                    shutil.copyfileobj(source, dest)
        # Exclusive creation, even if another process has created the destination.
        with staged.open('rb') as source, out.open('xb') as dest:
            shutil.copyfileobj(source, dest)
    summary = dict(
        source_accepted_events=expected, source_summary=source_summary,
        channel_C_event_pairs=len(struck), rows=2*len(struck), physics=physics,
        workers_requested=args.workers, workers_effective=workers, max_events=args.max_events,
        baryon_species={'proton': 2212, 'neutron': 2112},
        final_state='Original isFinal() and isHadron(); positive PDG IDs only',
        neutron_check='particleData.mayDecay(2112) must be false; no settings changed',
        rng='Original PYTHIA defaults; contiguous shard batches and sorted event IDs',
        arms=[{k: v for k, v in result.items() if k != 'path'} for result in results],
        csv=str(out), complete=True,
    )
    with summary_path.open('x') as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps({k: summary[k] for k in ['rows', 'channel_C_event_pairs', 'csv', 'complete']}, indent=2))


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Preflight and small repository manifest for the existing Fig. 3 A1–A3 pools."""
import argparse
import hashlib
import json
from pathlib import Path
import pandas as pd


def validate(source, expected):
    if not 0 < expected <= 9600000:
        raise ValueError('Accepted-event count must lie in [1,9600000]')
    report = dict(source_parent=str(source), accepted_events=expected, reuse_A1_A2=False, reuse_A3=False)
    editable = source/'editable_source_v1'
    if editable.exists():
        summary = json.loads((editable/'run_summary.json').read_text())
        if summary['n_accepted'] != expected:
            raise ValueError('Existing editable source count differs from requested count')
        count = sum(len(pd.read_parquet(p, columns=['event_id'])) for p in (editable/'events').glob('shard_*.parquet'))
        if count != expected:
            raise ValueError('Existing event shards disagree with run_summary')
        # Inspect physical acceptance using stored generator event columns.
        probe = pd.read_parquet(next((editable/'events').glob('shard_*.parquet')))
        if 'Q2' in probe and not probe.Q2.between(16,25,inclusive='neither').all():
            raise ValueError('Source Q2 acceptance differs from Fig. 3')
        report.update(reuse_A1_A2=True, editable_summary=summary)
    altered = source/'altered_100k_parquet'
    if altered.exists():
        summary = json.loads((altered/'run_summary.json').read_text())
        if summary['total_processed'] != expected or summary.get('surgery_version') != 'v2':
            raise ValueError('Require matching v2 source pool')
        if summary.get('seed') != 42 or summary.get('delta_px') != .4:
            raise ValueError('Source surgery settings differ from original Fig. 3 defaults')
        if not report['reuse_A1_A2']:
            raise ValueError('Altered input has no matching editable input')
        report.update(reuse_A3=True, altered_summary=summary)
    return report


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-parent', type=Path, required=True)
    ap.add_argument('--expected-events',type=int, required=True)
    ap.add_argument('--manifest',type=Path)
    ap.add_argument('--workers',type=int,default=2)
    a=ap.parse_args()
    report=validate(a.source_parent.expanduser().resolve(),a.expected_events)
    print(json.dumps({k:v for k,v in report.items() if not k.endswith('_summary')},indent=2))
    if a.manifest:
        repo=Path(__file__).resolve().parents[2]
        paths=[repo/'run_fig3_pi0.sh',repo/'run_fig3.sh',*Path(__file__).parent.glob('*pi0*.py')]
        report.update(workers=a.workers, generation_seed=12345, surgery_seed=42, pooled_balance_seed=42,
                      source_sha256={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        if a.manifest.exists():
            old=json.loads(a.manifest.read_text())
            if old['accepted_events']!=report['accepted_events'] or old['source_parent']!=report['source_parent']:
                raise ValueError('Existing configuration refers to different inputs')
            print('Preserving existing configuration manifest')
        else:
            a.manifest.parent.mkdir(parents=True,exist_ok=True)
            with a.manifest.open('x') as f: json.dump(report,f,indent=2)

if __name__=='__main__': main()

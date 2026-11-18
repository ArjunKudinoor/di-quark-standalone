#!/usr/bin/env python3
"""Reuse the pi0 Track A preflight; write only a new baryon provenance manifest."""
import argparse
import hashlib
import json
from pathlib import Path
from validate_fig3_pi0_inputs import validate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-parent', type=Path, required=True)
    ap.add_argument('--expected-events', type=int, default=9600000)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--manifest', type=Path)
    ap.add_argument('--require-editable', action='store_true')
    ap.add_argument('--require-altered', action='store_true')
    args = ap.parse_args()
    source = args.source_parent.expanduser().resolve()
    report = validate(source, args.expected_events)
    if args.require_editable and not report['reuse_A1_A2']:
        raise ValueError('Selected resume stage requires an existing editable source')
    if args.require_altered and not report['reuse_A3']:
        raise ValueError('Selected resume stage requires existing v2 altered data')
    print(json.dumps({k: v for k, v in report.items() if not k.endswith('_summary')}, indent=2))
    if args.manifest:
        repo = Path(__file__).resolve().parents[2]
        manifest = args.manifest.expanduser().resolve()
        if not manifest.is_relative_to(repo/'outputs'):
            raise ValueError('Manifest must be under repository outputs/')
        paths = [repo/'run_fig3_baryons.sh', *Path(__file__).parent.glob('*baryon*.py'),
                 Path(__file__).parent/'generate_dis_editable_source_parquet.py',
                 Path(__file__).parent/'transform_editable_all_altered_parquet.py',
                 Path(__file__).parent/'split_dis_sample_diquark_kick_v2.py',
                 Path(__file__).parent/'jet_hadron_observables_split_pi_pm.py']
        report.update(workers=args.workers, generation_seed=12345, surgery_seed=42,
                      pooled_balance_seed=42, source_sha256={
                          str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        if manifest.exists():
            old = json.loads(manifest.read_text())
            if old['source_parent'] != str(source) or old['accepted_events'] != args.expected_events:
                raise ValueError('Existing baryon configuration refers to different inputs')
            print('Preserving existing baryon configuration')
        else:
            manifest.parent.mkdir(parents=True, exist_ok=True)
            with manifest.open('x') as stream:
                json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()

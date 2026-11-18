#!/usr/bin/env python3
"""Smoke physics checks and exact reproduction using the unmodified original CLI."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import jet_hadron_observables_split_pi_pm as original
import jet_hadron_observables_split_pi_pm_pi0 as extended


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data-parent',type=Path,required=True)
    ap.add_argument('--summary-dir',type=Path,required=True)
    ap.add_argument('--figure-dir',type=Path,required=True)
    a=ap.parse_args()
    p4=lambda e,z:np.array([e,.3,.1,z])
    hadrons=[(211,p4(2,1)),(-211,p4(3,2)),(111,p4(4,3)),(111,p4(8,-3))]
    assert original.best_leading_charged_pion_breit(hadrons)[0]==-211
    assert extended.best_leading_pion_breit(hadrons)[0]==111
    assert extended.best_leading_pion_breit(hadrons)[1][0]==4
    for z in [-10.,-1.,1.,10.]:
        v=p4(abs(z)+1,z)
        kept=extended.discard_pions_outside_collider_eta([(211,v),(-211,v),(111,v)],np.eye(4),2,3.5)
        assert len(kept) in [0,3]
    proton,q=np.array([10.,0,0,10]),np.array([0.,0,0,-4])
    assert original.compute_observables_for_pion(p4(4,.01),np.array([.1,-.3]),proton,q)[-1]=='zLC_range'
    phi,*rest=original.compute_observables_for_pion(p4(4,3),np.array([.1,-.3]),proton,q)
    assert rest[-1]=='ok' and np.isclose(phi,np.pi/2)
    analysis=Path(__file__).parent
    common=['--altered-root',str(a.data_parent/'altered_100k_parquet'),
            '--editable-parent',str(a.data_parent),'--kinematics-preset','e5_ep41_x005_q2to4',
            '--beam-e','5','--beam-p','41','--dis-q-min','4','--dis-q-max','5',
            '--pion-eta-min','2.0','--pion-eta-max','3.5','--workers','2']
    old=a.data_parent/'charged_only_original_validation.csv'
    new=a.data_parent/'charged_only_pi0_validation.csv'
    report=a.summary_dir/'charged_only_driver_summary.json'
    for p in [old,new,report,a.summary_dir/'validation.json']:
        if p.exists(): raise SystemExit(f'Refusing to overwrite {p}')
    subprocess.run([sys.executable,str(analysis/'jet_hadron_observables_altered_parquet_filtered.py'),
                    *common,'--out-csv',str(old)],check=True)
    subprocess.run([sys.executable,str(analysis/'jet_hadron_observables_altered_parquet_filtered_pi0.py'),
                    *common,'--out-csv',str(new),'--charged-only','--summary-json',str(report)],check=True)
    x,y=pd.read_csv(old),pd.read_csv(new)
    pd.testing.assert_frame_equal(x,y[x.columns],check_dtype=False)
    # End-to-end extended output has exactly one category per source/arm.
    data=pd.read_csv(a.data_parent/'jet_hadron_pi_pm_pi0_observables.csv')
    assert not data.duplicated(['event_id','sample']).any()
    valid=data[data.ok]
    assert valid.pion.isin(['piplus','piminus','pi0']).all()
    assert (valid['pion_lab_eta'].gt(2)&valid['pion_lab_eta'].lt(3.5)).all()
    assert valid['pion_breit_pz_GeV'].gt(0).all()
    assert valid.zLC.between(.1,original.ZLC_MAX).all()
    assert valid.phi_hJ.between(0,np.pi).all()
    neutral=int(sum(valid.pion=='pi0'))
    assert neutral>0
    # Original style control remains inside the repo.
    a.figure_dir.mkdir(parents=True,exist_ok=True)
    subprocess.run([sys.executable,str(analysis/'plot_jet_hadron_pi_pm_observables.py'),
                    '--csv',str(old),'--figure-dir',str(a.figure_dir),'--layout','combined',
                    '--only-ok-rows','--combined-only','angle',
                    '--compare-pooled-altered-fractions','0,0.05,0.10','--pooled-balance-seed','42',
                    '--combined-filename-suffix','_charged_validation','--no-title'],check=True)
    result=dict(charged_original_rows=len(x),exact_common_fields_equal=True,
        original_and_extended_column_count=[len(x.columns),len(y.columns)],
        extended_rows=len(data),accepted_neutral=neutral,physics_checks_passed=True,
        original_source_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
        original_runner_sha256=hashlib.sha256((analysis.parents[1]/'run_fig3.sh').read_bytes()).hexdigest())
    with (a.summary_dir/'validation.json').open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()

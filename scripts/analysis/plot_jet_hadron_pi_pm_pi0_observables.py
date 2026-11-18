#!/usr/bin/env python3
"""Fig. 3-style pion plots using a same-SRC-fraction pi+ denominator."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plot_jet_hadron_pi_pm_observables import _truthy_ok_mask, subsample_altered_unchanged_for_pool
from jet_hadron_pi_pm_figures import JH, _ratio_series
from jet_hadron_pi_pm_pi0_figures import write_combined_pi_pm_pi0_angle_pdf, write_piplus_src_comparison_pdf


def read_successful_observables(csv):
    """Read only needed columns in chunks; retain all successful species for pooling.

    Preserve CSV row order so seeded subsampling matches the original plots.
    The completed source CSV was already checked for event pairing upstream.
    """
    parts = []
    for chunk in pd.read_csv(csv, usecols=['event_id', 'sample', 'pion', 'ok', 'phi_hJ'], chunksize=250000):
        parts.append(chunk.loc[_truthy_ok_mask(chunk)])
    if not parts:
        raise ValueError('No observable rows')
    return pd.concat(parts, ignore_index=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv',type=Path,required=True)
    ap.add_argument('--figure-dir',type=Path,required=True)
    ap.add_argument('--summary-dir',type=Path,required=True)
    ap.add_argument('--layout',choices=['combined'],default='combined')
    ap.add_argument('--only-ok-rows',action='store_true')
    ap.add_argument('--combined-only',choices=['angle'],default='angle')
    ap.add_argument('--compare-pooled-altered-fractions',default='0,0.05,0.10')
    ap.add_argument('--pooled-balance-seed',type=int,default=42)
    ap.add_argument('--combined-filename-suffix',default='')
    ap.add_argument('--no-title',action='store_true')
    ap.add_argument('--caption',default='')
    ap.add_argument('--piplus-only',action='store_true',
                    help='Make only the additional pi+ SRC plot/tables from an existing CSV.')
    a=ap.parse_args()
    fractions=[float(f) for f in a.compare_pooled_altered_fractions.split(',')]
    if fractions != [0., .05, .10]: ap.error('Use original 0%,5%,10% SRC fractions')
    a.figure_dir.mkdir(parents=True,exist_ok=True)
    a.summary_dir.mkdir(parents=True,exist_ok=True)
    outputs=[a.figure_dir/f'fig3_pi0_src{int(f*100)}.pdf' for f in fractions]
    csvnames=['histograms.csv','ratios.csv','tail_probabilities.csv','event_counts.csv','mixtures.csv']
    plus_output=a.figure_dir/'fig3_piplus_src_comparison.pdf'
    plus_names=['piplus_src_histograms.csv','piplus_src_ratios.csv','piplus_src_mixtures.csv','piplus_src_summary.json']
    targets=[plus_output]+[a.summary_dir/n for n in plus_names]
    if not a.piplus_only:
        targets+=outputs+[a.summary_dir/n for n in csvnames]+[a.summary_dir/'plot_summary.json']
    for p in targets:
        if p.exists(): raise SystemExit(f'Refusing to overwrite {p}')
    data=read_successful_observables(a.csv) if a.piplus_only else pd.read_csv(a.csv,low_memory=False)
    assert not data.duplicated(['event_id','sample']).any()
    if set(data['sample']) != {'altered','unchanged'}: raise ValueError('Require both analysis arms')
    if not a.piplus_only:
        groups=[set(data.loc[data['sample'].eq(s),'event_id']) for s in ['altered','unchanged']]
        if groups[0] != groups[1]: raise ValueError('Altered/unchanged source event IDs differ')
        counts=data.groupby(['sample','pion','reason'],dropna=False).size().reset_index(name='events')
    data=data.loc[_truthy_ok_mask(data)].copy()
    if not data.pion.isin(['piplus','piminus','pi0']).all(): raise ValueError('Unexpected pion species')
    if not data.phi_hJ.between(*JH.ANGLE_RANGE).all(): raise ValueError('Invalid angle')
    unc=data.loc[data['sample'].eq('unchanged')].copy()
    # Keep 0% pi+ only for the separate pi+ 5%/0% and 10%/0% comparison.
    baseline_piplus=unc.loc[unc.pion.eq('piplus'),'phi_hJ'].to_numpy()
    bins=np.linspace(*JH.ANGLE_RANGE,JH.ANGLE_BINS+1)
    width=bins[1]-bins[0]
    hrows=[]; rrows=[]; tails=[]; mixtures=[]
    plus_values={}; plus_hist=[]; plus_ratios=[]; plus_mixtures=[]
    for i,f in enumerate(fractions):
        if f == 0: pool,na,nu,got=unc.copy(),0,len(unc),0.
        else:
            pool,na,nu,got=subsample_altered_unchanged_for_pool(data,f,seed=a.pooled_balance_seed+i)
            if not na or not nu: raise ValueError('Insufficient rows to form SRC mixture')
        # Extract pi+ AFTER pooling all species, using precisely the same pool.
        values=pool.loc[pool.pion.eq('piplus'),'phi_hJ'].to_numpy()
        if not len(values): raise ValueError('No pi+ in SRC pool')
        plus_values[f]=values
        hist=np.histogram(values,bins)[0]
        density=hist/(len(values)*width)
        assert np.isclose(density.sum()*width,1.)
        plus_mixtures.append(dict(fraction=f,altered=na,unchanged=nu,achieved_fraction=got,piplus=len(values)))
        for j,n in enumerate(hist):
            plus_hist.append(dict(fraction=f,low=bins[j],high=bins[j+1],count=int(n),total=len(values),density=density[j]))
        if f:
            centers,ratio,err,valid=_ratio_series(baseline_piplus,values,bins)
            for j in range(len(centers)):
                plus_ratios.append(dict(numerator_fraction=f,denominator_fraction=0.,low=bins[j],high=bins[j+1],
                                        ratio=ratio[j],se=err[j],valid=bool(valid[j])))
        if a.piplus_only:
            print(f'Pi+ SRC {f:.0%}: altered={na}, unchanged={nu}, pi+={len(values)}')
            continue
        # All three species in this figure come from this same mixture.
        vp=plus_values[f]
        vm=pool.loc[pool.pion.eq('piminus'),'phi_hJ'].to_numpy()
        v0=pool.loc[pool.pion.eq('pi0'),'phi_hJ'].to_numpy()
        if min(len(vp),len(vm),len(v0))==0: raise ValueError('At least one species is empty')
        mixtures.append(dict(fraction=f,altered=na,unchanged=nu,achieved_fraction=got,
                             piplus=len(vp),piminus=len(vm),pi0=len(v0)))
        for species,values in [('piplus',vp),('piminus',vm),('pi0',v0)]:
            hist=np.histogram(values,bins)[0]
            density=hist/(len(values)*width)
            assert np.isclose(np.sum(density)*width,1.)
            for j,n in enumerate(hist):
                hrows.append(dict(fraction=f,species=species,low=bins[j],high=bins[j+1],
                                  count=int(n),total=len(values),density=density[j]))
            tail_count=int(np.sum(values>3*np.pi/4)); p=tail_count/len(values)
            tails.append(dict(fraction=f,species=species,count=tail_count,total=len(values),
                              probability=p,binomial_se=np.sqrt(p*(1-p)/len(values))))
        for species,values in [('piminus',vm),('pi0',v0)]:
            centers,ratio,err,valid=_ratio_series(vp,values,bins)
            for j,x in enumerate(centers):
                rrows.append(dict(fraction=f,numerator=species,denominator='same_fraction_piplus',denominator_fraction=f,
                                  low=bins[j],high=bins[j+1],ratio=ratio[j],se=err[j],valid=bool(valid[j])))
        write_combined_pi_pm_pi0_angle_pdf(vp=vp,vm=vm,v0=v0,fraction=f,output=outputs[i],caption=a.caption)
        print(f'SRC {f:.0%}: altered={na}, unchanged={nu}; wrote {outputs[i]}')
    write_piplus_src_comparison_pdf(values_by_fraction=plus_values,output=plus_output,caption=a.caption)
    for name,rows in [('piplus_src_histograms.csv',plus_hist),('piplus_src_ratios.csv',plus_ratios),
                      ('piplus_src_mixtures.csv',plus_mixtures)]:
        pd.DataFrame(rows).to_csv(a.summary_dir/name,index=False,mode='x')
    plus_summary=dict(source_csv=str(a.csv),normalization='Each pi+ curve separately integrates to one',
        fractions=fractions,bins=JH.ANGLE_BINS,pooled_balance_seed=a.pooled_balance_seed,
        selection='Original all-species successful-row pools; extract pi+ after subsampling',
        ratios='Normalized pi+(5%)/pi+(0%) and pi+(10%)/pi+(0%)',
        ratio_error='Original _ratio_series Poisson convention; overlap and normalization covariance ignored',
        streaming_successful_rows_only=a.piplus_only)
    with (a.summary_dir/'piplus_src_summary.json').open('x') as stream: json.dump(plus_summary,stream,indent=2)
    print(f'Wrote {plus_output}')
    if a.piplus_only:
        return
    for name,rows in [('histograms.csv',hrows),('ratios.csv',rrows),('tail_probabilities.csv',tails),('mixtures.csv',mixtures)]:
        pd.DataFrame(rows).to_csv(a.summary_dir/name,index=False,mode='x')
    counts.to_csv(a.summary_dir/'event_counts.csv',index=False,mode='x')
    summary=dict(rows_input=len(groups[0])*2,source_event_pairs=len(groups[0]),
        successful_rows=len(data),bins=JH.ANGLE_BINS,range=list(JH.ANGLE_RANGE),
        normalization='Each species separately integrates to one',
        reference='Pi+ from the same SRC pool as pi- and pi0 in each figure',
        ratios='(N_h_bin/N_h)/(N_plus_bin/N_plus)',
        ratio_error='R * sqrt(1/N_h_bin + 1/N_plus_bin); original _ratio_series function',
        empty_ratio_bins='Omitted, exactly as original',
        upper_panel='Original step curves without statistical bands',
        tail='phi_hJ > 3*pi/4; binomial errors for exploratory summary only',
        pooled_balance_seed=a.pooled_balance_seed,
        limitations='Original ratio errors ignore normalization and source-twin covariance; no model or detector uncertainty')
    with (a.summary_dir/'plot_summary.json').open('x') as stream: json.dump(summary,stream,indent=2)

if __name__=='__main__': main()

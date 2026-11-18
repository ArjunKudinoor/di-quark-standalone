#!/usr/bin/env python3
"""Plot normalized p/n, same-fraction n/p, and species-specific SRC comparisons."""
import argparse
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plot_jet_hadron_pi_pm_observables import _truthy_ok_mask, subsample_altered_unchanged_for_pool
from jet_hadron_pi_pm_figures import JH, _ratio_series
from jet_hadron_baryon_figures import (
    write_baryon_angle_pdf, write_baryon_src_comparison_pdf, DENSITY_YLIM, RATIO_YLIM,
)


def read_observables(path):
    """Keep original CSV ordering, including all successful species before pooling."""
    parts, outcomes, arm_counts, previous = [], Counter(), Counter(), {}
    columns = ['event_id', 'sample', 'baryon', 'ok', 'reason', 'phi_hJ']
    for chunk in pd.read_csv(path, usecols=columns, chunksize=250000):
        for sample, group in chunk.groupby('sample', sort=False):
            if sample not in ('altered', 'unchanged'):
                raise ValueError('Unexpected sample arm')
            ids = group.event_id.to_numpy()
            if not np.all(np.diff(ids) > 0) or (sample in previous and ids[0] <= previous[sample]):
                raise ValueError('Require increasing unique source IDs within each arm')
            previous[sample] = int(ids[-1])
            arm_counts[sample] += len(group)
        outcomes.update({(str(sample), str(reason)): int(n) for (sample, reason), n in
                         chunk.groupby(['sample', 'reason'], dropna=False).size().items()})
        parts.append(chunk.loc[_truthy_ok_mask(chunk)])
    if not parts or not arm_counts['altered'] or arm_counts['altered'] != arm_counts['unchanged']:
        raise ValueError('Missing or unequal source arms; use a completed A4 CSV')
    return pd.concat(parts, ignore_index=True), outcomes, arm_counts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--csv', type=Path, required=True)
    ap.add_argument('--figure-dir', type=Path, required=True)
    ap.add_argument('--summary-dir', type=Path, required=True)
    ap.add_argument('--observables-summary', type=Path, required=True)
    ap.add_argument('--compare-pooled-altered-fractions', default='0,0.05,0.10')
    ap.add_argument('--pooled-balance-seed', type=int, default=42)
    ap.add_argument('--caption', default='')
    args = ap.parse_args()
    fractions = [float(f) for f in args.compare_pooled_altered_fractions.split(',')]
    if fractions != [0., .05, .10]:
        ap.error('Use original 0%, 5%, 10% SRC fractions')
    repo = Path(__file__).resolve().parents[2]
    if not (args.figure_dir.resolve().is_relative_to(repo/'outputs')
            and args.summary_dir.resolve().is_relative_to(repo/'outputs')):
        ap.error('Figures and small tables must be inside repository outputs/')
    completion = json.loads(args.observables_summary.read_text())
    if not completion.get('complete') or Path(completion['csv']).resolve() != args.csv.resolve():
        raise ValueError('A4 completion summary does not match this CSV')
    outputs = [args.figure_dir/f'fig3_baryons_src{int(round(f*100))}.pdf' for f in fractions]
    data, outcomes, arms = read_observables(args.csv)
    if sum(arms.values()) != completion['rows'] or arms['altered'] != completion['channel_C_event_pairs']:
        raise ValueError('CSV row counts disagree with A4 completion summary')
    if not data.baryon.isin(['proton', 'neutron']).all() or not data.phi_hJ.between(*JH.ANGLE_RANGE).all():
        raise ValueError('Unexpected accepted baryon species or angle')
    unchanged = data[data['sample'].eq('unchanged')].copy()
    bins = np.linspace(*JH.ANGLE_RANGE, JH.ANGLE_BINS + 1)
    width = bins[1] - bins[0]
    hist_rows, ratio_rows, mixtures = [], [], []
    pools = []
    for i, fraction in enumerate(fractions):
        if fraction == 0:
            pool, na, nu, got = unchanged.copy(), 0, len(unchanged), 0.
        else:
            pool, na, nu, got = subsample_altered_unchanged_for_pool(
                data, fraction, seed=args.pooled_balance_seed+i)
            if not na or not nu:
                raise ValueError('Insufficient successful rows to form the requested SRC mixture')
        protons = pool.loc[pool.baryon.eq('proton'), 'phi_hJ'].to_numpy()
        neutrons = pool.loc[pool.baryon.eq('neutron'), 'phi_hJ'].to_numpy()
        if not len(protons) or not len(neutrons):
            raise ValueError('An SRC pool has no accepted p or n; no fallback selection or extra generation')
        pools.append((protons, neutrons))
        mixtures.append(dict(fraction=fraction, altered=na, unchanged=nu, achieved_fraction=got,
                             protons=len(protons), neutrons=len(neutrons)))
        for species, values in [('proton', protons), ('neutron', neutrons)]:
            counts = np.histogram(values, bins)[0]
            density = counts/(len(values)*width)
            if not np.isclose(density.sum()*width, 1.):
                raise RuntimeError('Histogram normalization failed')
            for j, n in enumerate(counts):
                hist_rows.append(dict(fraction=fraction, species=species, low=bins[j], high=bins[j+1],
                                      count=int(n), total=len(values), density=density[j]))
        centers, ratio, error, valid = _ratio_series(protons, neutrons, bins)
        for j in range(len(centers)):
            ratio_rows.append(dict(fraction=fraction, denominator='same_fraction_proton',
                                   denominator_fraction=fraction, low=bins[j], high=bins[j+1],
                                   ratio=ratio[j], se=error[j], valid=bool(valid[j])))
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.summary_dir.mkdir(parents=True, exist_ok=True)
    for f, (protons, neutrons), output in zip(fractions, pools, outputs):
        write_baryon_angle_pdf(protons=protons, neutrons=neutrons, fraction=f,
                               output=output, caption=args.caption)
        print(output)
    # Reuse the exact arrays above: no additional pooling or leading selection.
    src_ratio_rows = []
    comparison_outputs = {}
    for species, index in [('neutron', 1), ('proton', 0)]:
        values_by_fraction = {f: pair[index] for f, pair in zip(fractions, pools)}
        baseline = values_by_fraction[0.]
        baseline_counts = np.histogram(baseline, bins)[0]
        for fraction in [.05, .10]:
            numerator = values_by_fraction[fraction]
            numerator_counts = np.histogram(numerator, bins)[0]
            centers, ratio, error, valid = _ratio_series(baseline, numerator, bins)
            for j in range(len(centers)):
                src_ratio_rows.append(dict(
                    species=species, numerator_fraction=fraction, denominator_fraction=0.,
                    low=bins[j], high=bins[j+1], numerator_count=int(numerator_counts[j]),
                    numerator_total=len(numerator), denominator_count=int(baseline_counts[j]),
                    denominator_total=len(baseline), ratio=ratio[j], se=error[j], valid=bool(valid[j]),
                ))
        output = args.figure_dir/f'fig3_{species}_src_comparison.pdf'
        write_baryon_src_comparison_pdf(species=species, values_by_fraction=values_by_fraction,
                                        output=output, caption=args.caption)
        comparison_outputs[species] = str(output)
        print(output)
    for name, rows in [('histograms.csv', hist_rows), ('ratios.csv', ratio_rows), ('mixtures.csv', mixtures)]:
        pd.DataFrame(rows).to_csv(args.summary_dir/name, index=False, mode='w')
    pd.DataFrame([dict(sample=s, reason=r, events=n) for (s, r), n in sorted(outcomes.items())]).to_csv(
        args.summary_dir/'event_counts.csv', index=False, mode='w')
    pd.DataFrame(src_ratio_rows).to_csv(args.summary_dir/'src_comparison_ratios.csv', index=False, mode='w')
    summary = dict(source_csv=str(args.csv.resolve()), rows_input=sum(arms.values()),
                   accepted_rows=len(data), bins=JH.ANGLE_BINS, range=list(JH.ANGLE_RANGE),
                   normalization='Unit integral separately for p and n',
                   reference='Proton distribution from the same SRC fraction',
                   ratios='(N_n_bin/N_n)/(N_p_bin/N_p)',
                   ratio_error='Original R*sqrt(1/N_n_bin+1/N_p_bin); empty bins omitted',
                   limitations='Original errors omit normalization/source-twin covariance and model/detector effects',
                   pooled_balance_seed=args.pooled_balance_seed,
                   src_comparison_figures=comparison_outputs,
                   src_comparison_reference='Same species at 0% SRC, normalized to unit integral',
                   src_comparison_ratios='D_species(5%)/D_species(0%) and D_species(10%)/D_species(0%)',
                   src_comparison_empty_bins='Empty numerator or denominator omitted; CSV ratio/error NaN and valid=False',
                   src_comparison_uncertainty='Original Poisson convention; overlapping-pool covariance is not included',
                   density_ylim=list(DENSITY_YLIM), ratio_ylim=list(RATIO_YLIM))
    with (args.summary_dir/'plot_summary.json').open('w') as stream:
        json.dump(summary, stream, indent=2)


if __name__ == '__main__':
    main()

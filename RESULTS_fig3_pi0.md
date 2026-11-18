# Fig. 3 π0 reimplementation: completed results

The replacement follows the original A1–A5 pipeline and plotting conventions.
See [README_fig3_pi0.md](README_fig3_pi0.md) for exact commands, file descriptions,
physics details, and storage layout. No original tracked source files changed.

## Completed execution and validation

- Default executable smoke run: `./run_fig3_pi0.sh` succeeded; reused the existing
  1000-event pool and extracted both arms for 702 successful channel-C source IDs.
- Exact charged-only regression: the unmodified original driver and replacement
  in charged-only mode produced identical values for all 1404 rows and all 22
  original columns. Original plotting CLI generated the charged-only control.
- Focused physics checks passed for single leading-pion competition, target
  hemisphere, common eta acceptance, subsequent zLC rejection, and an independent
  transverse-angle calculation. The smoke extension selected 107 accepted π0.
- Intermediate run: 12×40000 accepted events, with A1 generation, A2 merging and
  A3 v2 surgery reused. No new DIS events were generated. Rehadronization covered
  all 334001 successful channel-C source events and their unchanged twins:
  668002 observable rows, including failures retained for accounting.
- Intermediate validation confirmed exact event pairing, one row per event/arm,
  original cuts and angular range, unit-integral histograms, and the exact
  original ratio and error formula. All configuration source hashes match.
- All three PDFs were rendered and visually checked against the original style.
  The 480000-event cap was checked with a request for 480012 events, which was
  rejected before writing output paths. `bash -n` passed for the new runner.
- `git diff --exit-code` passes: existing tracked files are unchanged.

## Figures and numerical data

Final repository figures:

- [0% SRC](outputs/figures/fig3_pi0_src0.pdf)
- [5% SRC](outputs/figures/fig3_pi0_src5.pdf)
- [10% SRC](outputs/figures/fig3_pi0_src10.pdf)

Small tables and manifests are in `outputs/analysis/fig3_pi0_480000/`:
`histograms.csv`, `ratios.csv`, `tail_probabilities.csv`, `mixtures.csv`,
`event_counts.csv`, `configuration.json`, `observables_summary.json`,
`plot_summary.json`, and `validation.json`.

The analogous smoke products are in `outputs/analysis/fig3_pi0_1000/` and
`outputs/figures/fig3_pi0_1000/`. Console logs are
`outputs/analysis/fig3_pi0_smoke_run.log`, `fig3_pi0_validation.log`, and
`fig3_pi0_intermediate_run.log`.

Large intermediate observable records are outside the repository:

```
~/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_480000_pi0/merged/jet_hadron_pi_pm_pi0_observables.csv
```

The neighboring `editable_source_v1` and `altered_100k_parquet` links refer to
unchanged original intermediate data. New source code, final PDFs, small CSVs,
configs and documentation are all inside the checkout. No original Fig. 3 PDF
or data product was overwritten.

## Physics interpretation

The neutral large-angle distribution is broadly similar to the negative-pion
one. For the exploratory integrated tail **φhJ > 3π/4**, probabilities are
(percent ± one binomial standard error):

| SRC pool | Shared unchanged π+ | π− | π0 |
|---|---:|---:|---:|
| 0% | 52.34 ± 0.27 | 52.29 ± 0.27 | 52.78 ± 0.26 |
| 5% | 52.34 ± 0.27 | 54.54 ± 0.26 | 54.28 ± 0.25 |
| 10% | 52.34 ± 0.27 | 56.39 ± 0.24 | 55.93 ± 0.25 |

At 5% and 10%, neutral central values lie between the charged tail probabilities.
The neutral-minus-negative differences are only −0.25 and −0.46 percentage
points; independent-error estimates are approximately 0.36 and 0.35 percentage
points. An ordering relative to π− is therefore unresolved with these statistics.
At 0%, the small neutral excess over both charged values is consistent with
fluctuations at this level. Individual-bin ordering fluctuates more than the
integrated tail. No general pointwise statement that π0 lies between π+ and π−
is established.

The probabilities summarize the extended leading-pion selection, including
competition with π0. Charged distributions can differ from the original
charged-only selection. Successful selections before pooling are:

| Arm | π+ | π− | π0 |
|---|---:|---:|---:|
| Unchanged | 34480 | 34830 | 36484 |
| Altered | 3366 | 20324 | 12277 |

There are 141761 successful rows, including 48761 π0 selections. Six PYTHIA
reinjection failures and 390 rows failing the original reevaluated x/Q window
are retained as failures and excluded, using the original procedure. The latter
counts are 195 per arm; accepted source kinematics are not replaced by a new
analysis cut. All successful rows pass the required cuts.

Mixtures contain (unchanged,altered)=(105794,0), (105794,5568), and
(105794,11755); achieved altered fractions are 0, 0.0499991020, and
0.1000008507. As in the original code, these fractions describe successful
observable rows. The blue π+ reference stays unchanged across all figures.

The plotted errors are **exactly the original Poisson ratio convention**,
`R sqrt(1/N_h_bin + 1/N_plus_bin)`. They omit normalization covariance,
shared-source/twin covariance and detector/model uncertainties. The tail
binomial errors and independent-error differences above are exploratory
summaries, with the same covariance limitation. No bootstrap analysis or
precision-based event-count increase was introduced. These generator-level
results do not establish detector reconstruction feasibility.

## Previous implementation removed and uncertain files preserved

The old `scripts/followup/pi0/` directory and verified products of its
`followup_pi0/smoke_20261019` and `followup_pi0/intermediate_20261019` runs were
removed (250 primary files plus six temporary previews/logs). There were no
previous tracked-file modifications to revert. The deletion audit is
`outputs/analysis/fig3_pi0_cleanup.json`.

Only the following old metadata files remain. Their creation or subsequent
changes cannot be attributed exclusively to the previous implementation, so
none was deleted:

- `~/Data/dis_q_bins_e5_ep41_standalone/followup_pi0/.DS_Store`
- `~/Data/dis_q_bins_e5_ep41_standalone/followup_pi0/smoke_20261019/.DS_Store`
- `~/Data/dis_q_bins_e5_ep41_standalone/followup_pi0/intermediate_20261019/.DS_Store`

All new files are listed in the README; the only other additions are these
new documentation files and generated products under `outputs/`. No commit
or push was performed.

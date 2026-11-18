# Fig. 3 with a neutral-pion extension

This follows `run_fig3.sh`, with parallel scripts beside its existing analysis
scripts. Original source files are unchanged. Large event/observable records
live outside the checkout; final figures, small CSV tables, run configurations,
and logs live under the repository's existing `outputs/` directory.

## Run from the repository root

```bash
# Default smoke configuration: 2 chunks × 500 accepted source events.
./run_fig3_pi0.sh

# Intermediate configuration: 12 chunks × 40000 accepted source events.
./run_fig3_pi0.sh --events-per-chunk 40000 --chunks 12 --gen-jobs 4 --workers 4

# Paper-like configuration: 12 chunks × 800000 = 9.6 million accepted events.
./run_fig3_pi0.sh --events-per-chunk 800000 --chunks 12 --gen-jobs 4 --workers 4

# Resume at the same A1–A5 stages as the original runner.
./run_fig3_pi0.sh --events-per-chunk 40000 --chunks 12 --workers 4 --start A4
```

### Add the π+ comparison to an existing 9.6-million-event analysis

```bash
./run_fig3_pi0.sh --events-per-chunk 800000 --chunks 12 --piplus-only
```

This reads the existing
`~/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_9600000_pi0/merged/jet_hadron_pi_pm_pi0_observables.csv`
and runs only histogramming and plotting. It does not generate DIS events,
perform surgery, or rehadronize. It also leaves all existing π0 figures and
tables untouched. If the CSV is missing, it stops instead of generating events.
The large CSV is read in chunks with only the necessary columns; unsuccessful
rows are discarded before retaining data in memory. All successful pion species
are kept for pooling, with π+ extracted only after the original subsampling.

The new outputs are:

- `outputs/figures/fig3_pi0_9600000/fig3_piplus_src_comparison.pdf`
- `outputs/analysis/fig3_pi0_9600000/piplus_src_histograms.csv`
- `outputs/analysis/fig3_pi0_9600000/piplus_src_ratios.csv`
- `outputs/analysis/fig3_pi0_9600000/piplus_src_mixtures.csv`
- `outputs/analysis/fig3_pi0_9600000/piplus_src_summary.json`

The upper panel shows **unit-normalized π+ distributions at 0%, 5%, and 10%**
SRC. The lower panel shows the normalized 5%/0% and 10%/0% π+ shape ratios.
The original 40 bins, pool seeds, figure style and Poisson ratio-error formula
are retained. These pools overlap strongly, so the inherited errors ignore
the correlation between numerator and denominator; they are not correlated
uncertainty estimates. No absolute yields are plotted.

Normal A5 runs now produce this comparison in addition to the original three
π0 figures. `--piplus-only` automatically selects A5 and makes just the new
outputs, which also refuse to overwrite existing files. To repeat a plot into
fresh directories, call the plotting script directly with `--piplus-only` and
fresh `--figure-dir` / `--summary-dir` paths.

### Plot all species using π+ from the matching SRC mixture

```bash
./run_fig3_pi0.sh --events-per-chunk 800000 --chunks 12 --start A5
```

This uses the existing observable CSV and does not generate events or
rehadronize. Each of the 0%, 5%, and 10% figures now takes **all three pion
distributions from that fraction's pool**. Both ratio denominators use the
π+ distribution at that same fraction. The blue legend explicitly labels
π+ with its percentage. Each distribution still has unit integral.

New figures go to `outputs/figures/fig3_pi0_9600000/`; your renamed
`outputs/figures/fig3_pi0_9600000_0_piplus_baseline/` remains untouched.
The corresponding new numerical tables go to
`outputs/analysis/fig3_pi0_9600000/matched_piplus/` so the old shared-baseline
tables are preserved as well. Existing destination files are never overwritten.
The separate π+ SRC-comparison figure continues to show 5%/0% and 10%/0%.

The runner uses an activated Python environment if present. Otherwise it uses
the checkout's `.venv`. If `pythia8` is missing from its search path, it tries
the existing sibling `../software/pythia8315/lib` binding. An explicit
`PYTHONPATH` remains effective. Dependencies are the same as the original
runner: numpy, pandas, pyarrow, matplotlib, tqdm, PYTHIA Python bindings, and
macOS `caffeinate`. No dependencies are installed automatically.

The original options (`--events-per-chunk`, `--chunks`, `--gen-jobs`,
`--workers`, `--start`) and A1–A5 order are retained. Counts over 9,600,000 are
rejected before output creation. The existing source pool is checked against
its manifests and event-shard counts. Where available, the runner reuses A1
source generation, A2 merging, and A3 v2 surgery through input-directory
symlinks; the analysis never writes into those input directories. Without
reusable inputs, the original generation/merge/surgery programs are used.
Generation seeds remain `12345 + chunk`; surgery and pool seeds remain 42.
Rehadronization retains the original PYTHIA default RNG and worker partition;
worker count is therefore part of the reproducible configuration.

Existing observable CSVs are preserved. Final plots and small tables refuse
to overwrite existing files. To replot manually into fresh destinations:

```bash
.venv/bin/python scripts/analysis/plot_jet_hadron_pi_pm_pi0_observables.py \
  --csv "$HOME/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_480000_pi0/merged/jet_hadron_pi_pm_pi0_observables.csv" \
  --figure-dir outputs/figures/pi0_replot \
  --summary-dir outputs/analysis/pi0_replot \
  --only-ok-rows --compare-pooled-altered-fractions 0,0.05,0.10 \
  --pooled-balance-seed 42
```

## Files and paths

Intermediate execution outputs:

- `outputs/figures/fig3_pi0_src0.pdf`
- `outputs/figures/fig3_pi0_src5.pdf`
- `outputs/figures/fig3_pi0_src10.pdf`
- `outputs/analysis/fig3_pi0_480000/`: configuration and observable-count summary;
  new histogram/ratio CSVs, mixture sizes and broad-tail probabilities are in
  its `matched_piplus/` subdirectory. Earlier baseline tables stay in place.
- `~/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_480000_pi0/merged/`:
  large observable CSV plus links to original editable and altered event data.

The 1000-event smoke run instead writes figures to
`outputs/figures/fig3_pi0_1000/`, summaries to
`outputs/analysis/fig3_pi0_1000/`, and large records to the analogous
`Q2_16_25__Q_4_5_1000_pi0/merged/` directory. This prevents smoke figures from
replacing intermediate figures. Original `signals*.pdf` files are preserved.

The paper-like run writes figures to `outputs/figures/fig3_pi0_9600000/`,
small run summaries to `outputs/analysis/fig3_pi0_9600000/` and new plot
tables to its `matched_piplus/` subdirectory, and large records to
`~/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_9600000_pi0/`.
The default remains the 1000-event smoke configuration; paper-like statistics
require the explicit command above.

New code:

- `run_fig3_pi0.sh`: close copy of the original root runner, with reuse,
  separated destinations and the study's event cap.
- `scripts/analysis/jet_hadron_observables_split_pi_pm_pi0.py`: explicit copies
  of the three selection-dependent helpers; imports the original kinematics,
  reinjection, color, zLC and angular helpers unchanged.
- `scripts/analysis/jet_hadron_observables_altered_parquet_filtered_pi0.py`:
  copy of the original channel-C driver, retaining its shard workers, ordering
  and pairing. Adds a charged-only validation option, output guards, candidate
  diagnostics and a small repository summary.
- `scripts/analysis/jet_hadron_pi_pm_pi0_figures.py`: the original two-panel
  rendering structure plus the neutral curve/ratio; imports original curve,
  ratio and axis-limit helpers.
- `scripts/analysis/plot_jet_hadron_pi_pm_pi0_observables.py`: original pool
  subsampling and new figure/table export.
- `scripts/analysis/validate_fig3_pi0_inputs.py`: bounded-input preflight and
  lightweight provenance manifest.
- `scripts/analysis/validate_fig3_pi0.py`: focused physics checks and exact
  charged-only CLI comparison against the unmodified original program.

## Physics and figure conventions

This extends [Peng, Rajagopal and Terry, arXiv:2606.17133](https://arxiv.org/abs/2606.17133)
using the repository's implementation of Fig. 3. The original cuts, frames,
selection order and SRC prescription are preserved:

1. Reuse the 5×41 GeV DIS sample: x>0.05, 16<Q²<25 GeV², struck u quark.
2. Keep successful channel-C v2 alterations and their unchanged source twins.
   Surgery remains `(ud) → d + (us) + sbar`, with longitudinal fractions
   x, 0.99−2x, 0.01, rejection above x=0.495, and the original balanced px
   kick with magnitude `0.4 sin(theta)` GeV and uniform theta on [0,π].
3. Use the original rehadronization and beam-to-Breit transforms. Apply strict
   **lab** 2<η<3.5 to eligible pions, then select **one** highest-**Breit-energy**
   pion with Breit pz>0 across π+, π− and π0. Apply the original zLC cut only
   after selecting the winner; no lower-energy fallback is introduced.
4. Use the outgoing struck-quark transverse momentum as jet proxy and the
   original angle on [0,π]. The exact zLC implementation accepts 0.1 through
   1+1e−6; it is unchanged from the source code.

π0 decays before the original final-hadron scan. The extension reads its parent
four-vector from the same PYTHIA event record, excluding intermediate copies
with π0 daughters. Charged pions still come from final hadrons. Decays, shower
cards, fragmentation and RNG consumption are unchanged; no photon reconstruction
or separate leading-pion selection is performed. The legacy diagnostic column
`n_final_hadrons` includes retained neutral parents after the extension.
`pion_breit_energy_GeV` and `pion_breit_pz_GeV` record the selected candidate.

Pool subsampling is the original function, after successful-observable cuts
and before species splitting (seeds 42,43,44). Consequently 0%,5%,10% describe
the altered fraction of the successful pool, as in the original plotting
code, rather than all generated DIS events. Paired source twins may both occur
in the pool. Each arm still selects at most one pion. Charged shapes change
when π0 wins the competition.

**The blue π+, red π− and green π0 curves all come from each fraction's pool.**
The π−/π+ and π0/π+ ratios use a same-fraction π+ denominator. This extends
the original plotting convention, which used a shared unchanged π+ reference;
the original repository plotter is unchanged. Each species
integrates to one separately. Ratios divide normalized shapes, not raw yields.
The original 40 bins on [0,π] are used.

The original 6×4.2-inch canvas, step curves, fonts, math labels, 1.5-point
curve widths, caption location, legend placement, inward ticks, and lower
strip (32%, pad 0.07) are retained. All upper panels now use fixed y-limits
**0.0 to 1.1**, and all ratio panels use **0.8 to 1.3**, including the separate
π+ SRC comparison.
The extra species uses
green; π+ remains blue and π− red. As in the original, the upper panel has no
error bands; the ratio panel uses points with caps. Ratio errors are the
**original** `R sqrt(1/N_h_bin + 1/N_plus_bin)` formula. Bins with an empty
numerator or denominator are omitted. These errors do not account for
normalization covariance or shared source twins, and no detector/model errors
are included. No bootstrap uncertainty scheme is substituted.

`tail_probabilities.csv` summarizes φhJ>3π/4 with binomial standard errors
for exploratory interpretation; the plotted errors remain the original ratio
formula. A central value between charged curves is not sufficient to establish
an ordering at intermediate statistics.

## Validation

After the default smoke run, reproduce the original charged-only selection
with two independent CLIs and compare every common CSV field:

```bash
export PYTHONPATH="$HOME/MIT/software/pythia8315/lib${PYTHONPATH:+:$PYTHONPATH}"
export MPLCONFIGDIR="${TMPDIR:-/tmp}/diquark-fig3-pi0-mpl"
.venv/bin/python scripts/analysis/validate_fig3_pi0.py \
  --data-parent "$HOME/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_1000_pi0/merged" \
  --summary-dir outputs/analysis/fig3_pi0_1000 \
  --figure-dir outputs/figures/fig3_pi0_1000/charged_control
```

Validation refuses to overwrite its previous control files. The completed
smoke comparison matches all 1404 rows across all 22 original columns, and
includes tests for species competition, hemisphere, common eta acceptance,
zLC rejection and an independent orthogonal-vector angle check. Its original
charged-only control figure is produced by the unmodified original plot CLI.

## Cleanup of the previous implementation

The previous `scripts/followup/pi0/` framework and its two dated external run
products were removed: 250 verified files, with no tracked original source
changes to undo. The exact deletion audit is
`outputs/analysis/fig3_pi0_cleanup.json`. Original shared Track A inputs were
preserved. Three `.DS_Store` files of uncertain origin were left untouched;
their exact paths are recorded in that audit. The old directories survive only
to hold those preserved metadata files. No commit or push is performed.

Completed numerical results, validation evidence, and cleanup details are in
[RESULTS_fig3_pi0.md](RESULTS_fig3_pi0.md).

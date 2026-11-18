# Fig. 3 proton/neutron follow-up — implementation only

All files are new. No original or pi0 source, data, figure or numerical output
is modified. This code has **not been executed, syntax-tested, simulated or
plotted** as part of its implementation. Commands below are for later use.

## Run later from the repository root

```bash
# Default: 12 x 800000 = 9600000 accepted DIS events; four workers/jobs.
# Reuse existing original/pi0 partons; perform baryon A4 extraction and A5 plots.
./run_fig3_baryons.sh

# Explicitly select the existing paper-scale pi0 parton pool and start at A4.
./run_fig3_baryons.sh --start A4 \
  --source-root "$HOME/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_9600000_pi0"

# Plot a completed baryon CSV only; no generation, surgery or rehadronization.
./run_fig3_baryons.sh --start A5

# Optional small run later, not performed during implementation.
./run_fig3_baryons.sh --events-per-chunk 500 --chunks 2 --gen-jobs 2 --workers 2
```

The familiar `--events-per-chunk`, `--chunks`, `--gen-jobs`, `--workers`, and
`--start A1|A2|A3|A4|A5` options are retained. The new `--source-root` argument
selects an existing root containing `merged/`. Maximum supported accepted-event
count is 9600000; the default is explicitly paper-scale, unlike the pi0 runner's
smoke default. The script is executable and uses the same Python environment,
existing local PYTHIA binding lookup, `caffeinate`, console logging and
`set -euo pipefail` convention. Bash arrays used with `set -u` are nonempty at
their expansion sites, including on macOS Bash 3.2.

## Reuse, stages and seeds

The inspected existing paper input is:

```
~/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_9600000_pi0/merged/
```

Its manifests record 9600000 accepted source events in 960 shards, merged
from twelve chunks; v2 surgery used seed 42 and kick parameter 0.4 GeV, with
6675716 successful channel-C alterations. No event files were opened or
analysis run during this task; only source and JSON/XML metadata were read.

The runner searches matching-count pi0, original, and then baryon roots for
reusable input, with the existing intermediate spelling supported for 480000.
It validates source counts and surgery settings before linking the existing
`editable_source_v1` and `altered_100k_parquet` input directories into the new
baryon root. No simulation writes through those links.

- A1: if no merged input exists, use the original DIS generator, chunk seeds
  `12345+w`, offsets `w*N_CHUNK`, 5x41 GeV beams, phase-space Q2 minimum 4,
  x>0.05, 16<Q2<25, and no generation kick.
- A2: reuse merged editable input or call the original chunk merger.
- A3: reuse the original/pi0 v2 alteration or call original surgery with
  `--surgery-version v2 --seed 42`. All SRC modeling parameters are unchanged.
- A4: rehadronize successful channel-C partons and their unchanged twins,
  now selecting a leading p/n. The pion observable CSV cannot supply baryon
  candidates, so A4 must run once for baryons. The shower/fragmentation cards
  and original default PYTHIA RNG are reused. Parallel workers retain the
  original contiguous shard partition and sorted event order, with separate
  altered and unchanged pools. Serial mode retains one RNG across both arms.
- A5: use the existing successful-row pooling function with seeds 42,43,44,
  then split each pool into its selected p/n species.

Worker count is part of the stochastic reproducibility configuration, as in
the pi0 implementation. At paper scale A4 must still process both versions of
each successful channel-C source event; that is not additional DIS generation.
To bound output-row memory, each worker streams newly-created temporary CSV
batches under the external baryon data root. Batches are concatenated in the
same sample/shard/event order used by the original driver. Only those newly
created temporary files are automatically cleaned up, not existing data.

## Physics and neutron convention

The helper follows the original/pi0 event processor's cut order and imports
the original beam transforms, color checks, final-hadron collector, light-cone
observable and angle calculation unchanged:

1. Apply the same DIS x/Q selection and color checks, then rehadronize.
2. Apply strict proton-forward **lab** 2<eta<3.5 to eligible p/n candidates
   using the original inverse Breit transformation and beam-axis flip.
3. Choose one maximum-**Breit-energy** candidate among **PDG +2212 and +2112**
   with Breit pz>0. There is no independent leading selection by species.
4. Apply the original zLC check after choosing the winner, with no fallback
   to a lower-energy baryon. The original code accepts 0.1 <= zLC <= 1+1e-6.
5. Use the same outgoing struck-u proxy and phi_hJ in [0,pi]. No new pT cuts.

Antiprotons and antineutrons are excluded, as only the positive PDG IDs were
requested. Other baryons, pions and nuclei do not compete for this selection.
Generated final-state nucleons from the original decay chains are included;
incoming beam-record protons do not pass the final-state collector.

**Neutron compatibility was checked by reading the installed PYTHIA 8.315
source, not by executing PYTHIA.** `ParticleData.xml` gives neutron lifetime
`tau0=2.65500e14` mm/c (id 2112, lines near 3500). In
`src/ParticleData.cc`, `ParticleDataEntry::MAXTAU0FORDECAY` is 1000 mm/c, and
`setDefaults()` sets `mayDecay` only below that threshold (lines near 105/127).
Thus default neutrons are stable even though a beta-decay channel is listed.
The original `isFinal() and isHadron()` collector includes them without a
charge/visibility requirement. No pre-decay reconstruction or decay toggles
are needed. The new builder checks `particleData.mayDecay(2112)` at runtime
and stops if another installation/configuration enables neutron decay.
It never silently changes simulation settings. This runtime guard has not
been executed during implementation.

## Figures, ratios and storage

### Species-specific SRC comparisons

Rerunning A5 now writes two additional figures from the completed baryon CSV:

```bash
./run_fig3_baryons.sh --start A5
```

- `outputs/figures/fig3_baryons_9600000/fig3_neutron_src_comparison.pdf`
- `outputs/figures/fig3_baryons_9600000/fig3_proton_src_comparison.pdf`

Each upper panel overlays that species at 0%, 5%, and 10% SRC, with a separate
unit-integral normalization for each curve. The lower panels show
`R_n^(5%/0%)` and `R_n^(10%/0%)`, or the corresponding proton ratios.
Both figures reuse the **exact arrays** used for the three existing p/n figures;
there is no separate resampling, leading-hadron selection, or event generation.
The seeds, 40 angular bins, colors (blue/purple/red for 0/5/10%), step curves,
two-panel layout and Poisson ratio errors follow the pi+ comparison convention.
Upper axes use [0,3.5] and ratio axes [0.8,1.6].

The same-species ratios use the original ratio helper: normalized numerator
shape divided by the normalized 0% shape, with
`R*sqrt(1/N_numerator_bin + 1/N_baseline_bin)`. Zero-count baseline bins are
omitted; zero-count numerator bins are also omitted to retain the original
uncertainty convention. Such entries have `valid=False` and NaN ratio/error
in `outputs/analysis/fig3_baryons_9600000/plots/src_comparison_ratios.csv`,
which also saves bin counts and totals. Existing `histograms.csv` contains
all six species/fraction distributions, so their densities are not duplicated.
The SRC pools share events; these inherited error bars do not include that
correlation or normalization covariance.

A5 replaces the five baryon PDFs and small plotting tables/summary when rerun,
while continuing to protect event data and the large observable CSV. These
code additions have not been executed or tested during implementation.

The code follows the latest pi0 plotting style: 6x4.2 inches, blue proton and
red neutron step curves, fraction-labelled legends, original caption/fonts,
inward ticks, 32% lower strip with pad 0.07, points with caps for ratios,
40 bins on [0,pi], and each species normalized to unit integral.

For each 0%,5%,10% pool, the denominator of n/p is the **proton distribution
from that same fraction**. Pooling occurs after successful-observable cuts
and before species splitting. Fractions therefore describe successful rows,
as in the original plot code; both twins of a source may enter the pool.

Ratios and errors call the original helper directly:
`(N_n_bin/N_n)/(N_p_bin/N_p)`, with error
`R*sqrt(1/N_n_bin+1/N_p_bin)`. Empty numerator/denominator bins are omitted.
These inherited errors do not include normalization or source-twin covariance,
nor model/detector uncertainties. The baryon upper-panel range is [0,3.5],
while the baryon ratio range is [0.8,1.6]. No pion plot
limits are changed. Baryon shapes have not yet been calculated;
if values lie outside these ranges, numerical CSVs still retain them.
No detector charge-tagging or neutron reconstruction model is introduced.

New outputs, only when run later:

- `outputs/figures/fig3_baryons_9600000/fig3_baryons_src0.pdf`
- `outputs/figures/fig3_baryons_9600000/fig3_baryons_src5.pdf`
- `outputs/figures/fig3_baryons_9600000/fig3_baryons_src10.pdf`
- `outputs/figures/fig3_baryons_9600000/fig3_neutron_src_comparison.pdf`
- `outputs/figures/fig3_baryons_9600000/fig3_proton_src_comparison.pdf`
- `outputs/analysis/fig3_baryons_9600000/configuration.json` and
  `observables_summary.json`; small plot tables and `plot_summary.json` under
  its `plots/` subdirectory.
- Large CSV and event-record links under
  `~/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_9600000_baryons/merged/`.

The repository's actual spelling is `outputs/figures/`, which is used instead
of introducing `output/figures/`. Every destination is baryon-specific.
Existing event data, the large A4 observable CSV, its completion summary and
mismatched configurations are protected from overwrites. A5 replaces its five
baryon PDFs and small plotting tables/summary, so rerunning
`./run_fig3_baryons.sh --start A5` applies plot-style changes to existing results.
Plotting requires a completed A4 summary. Missing
or empty-species pools cause an error rather than automatic extra generation
or an alternative leading-hadron definition.

## New files and implementation status

1. `run_fig3_baryons.sh` — executable A1--A5 runner, paper defaults and reuse.
2. `scripts/analysis/jet_hadron_observables_split_baryons.py` — p/n selection.
3. `scripts/analysis/jet_hadron_observables_altered_parquet_filtered_baryons.py`
   — paired rehadronization and streaming CSV extraction.
4. `scripts/analysis/validate_fig3_baryons_inputs.py` — reused preflight and
   new provenance manifests.
5. `scripts/analysis/jet_hadron_baryon_figures.py` — Fig. 3-style p/n rendering.
6. `scripts/analysis/plot_jet_hadron_baryon_observables.py` — pooled analysis,
   same-fraction ratios, plots and small CSVs.
7. `README_fig3_baryons.md` — this documentation.

No existing files were changed. No code, tests (including syntax checks),
simulations, rehadronization or plotting were executed. No event data or
figures were produced. Runtime correctness is therefore unverified. No
commit or push was performed.

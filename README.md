# Di-quark PYTHIA (standalone)

Monte-Carlo tooling to reproduce the two PYTHIA 8 figures from

> **Short-Range Correlations Between Partons in a Proton**
> Jen-Chieh Peng, Krishna Rajagopal, John Terry — MIT-CTP/6041, [arXiv:2606.17133](https://arxiv.org/abs/2606.17133)

The paper proposes an Electron-Ion Collider (EIC) observable to search for **partonic short-range
correlations (SRCs)** — in particular attractive **ud "good-diquark" pairs** — inside the proton.
In deep-inelastic scattering a struck **u** quark forms a current-region jet (tagged statistically
by a leading **π⁺**), while its SRC partner **d** quark emerges in the target region and fragments
into a **π⁻**. A ud SRC makes the jet and the π⁻ preferentially **back-to-back** in the Breit-frame
azimuthal angle **φ_hJ** (an enhancement near φ_hJ ∼ π), which shows up as a deviation of the ratio
**R_{π⁻/π⁺}(φ_hJ)** from unity.

## What this package produces

| Figure in paper | Output | Track | Description |
|-----------------|--------|-------|-------------|
| Fig. 2 | `Detectors.pdf` | **B** | Lab pseudorapidity η_h of the leading target pion for five EIC beam settings, overlaid with Central / B0 / Forward detector acceptance bands |
| Fig. 3 | `signals.pdf`   | **A** | Breit-frame φ_hJ for shared π⁺ vs π⁻ at 0% / 5% / 10% injected-SRC fraction, plus the R_{π⁻/π⁺} ratio strip |

The two tracks use **different generators, data layouts, and plot scripts** and must not be mixed.

## Physics settings (as in the paper)

- Beams for the signal figure: **Eₑ × E_p = 5 × 41 GeV²** (Track B also uses 5×100, 10×100, 10×275, 18×275).
- Kinematic cuts: **4 GeV < Q < 5 GeV**, **x > 0.05** (Eq. 1); central-detector region **2 < η_h < 3.5** (Eq. 2).
- Target pion: most energetic charged pion in the Breit target hemisphere (p_z^Breit > 0), with light-cone fraction **z_LC > 0.1**.
- SRC "surgery" (v2): replace the remnant `(ud) → d + (us) + s̄` to keep a valid PYTHIA colour dipole, assign longitudinal
  fractions **x**, **1 − 2x − 0.01**, **0.01** to `d`, `(us)`, `s̄`, then apply a **|k| = 0.4 GeV** back-to-back kick to the
  struck u and remnant d, in **5%** or **10%** of events, before re-hadronizing.

## Repository layout

```
src/diquark/            # library: path resolution, raw-event analysis
scripts/generation/     # Track B raw-event generation (generate_events_raw.py)
scripts/analysis/       # Track A pipeline + both plot scripts
AGENT_INSTRUCTIONS.md   # exact, copy-pasteable commands for both tracks
clean_tree.sh           # safe cleanup: reduce the repo to its source manifest
```

## Prerequisites

- Python 3.9+
- `numpy`, `pandas`, `pyarrow`, `matplotlib`, `tqdm`
- **PYTHIA 8** Python bindings (`import pythia8`) — required for generation and Track A reinjection

## Data location (important — do not melt your laptop)

Generation produces **thousands of shard / Parquet files** (hundreds of MB to several GB). These must live
**outside the repository and outside any cloud-synced or editor-indexed folder**.

- By default all data goes to **`~/Data/Di-quark-pythia-standalone-nosync/`** (the `-nosync` suffix marks it as
  "do not cloud-sync"). Override with `DIQUARK_DATA_ROOT`.
- **Never** point the data root under `~/Documents` or `~/Desktop` (iCloud-synced on macOS). This repo itself lives
  under `~/Documents`, so never let data be written inside the repo tree — iCloud + editor indexing on tens of
  thousands of files can thrash the machine. `src/diquark/paths.py` emits a warning if the root resolves under
  those locations.

## Quickstart

Run everything from the repository root. Values below are **SMOKE** scale (minutes); see
[`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) for paper-scale numbers and the full argument lists.

### Track B — `Detectors.pdf`

```bash
# 1) generate five-beam hadronized shards
python3 scripts/generation/generate_events_raw.py \
  --labels ETA_XQ_5x41,ETA_XQ_5x100,ETA_XQ_10x100,ETA_XQ_10x275,ETA_XQ_18x275 \
  --n_events 100000

# 2) plot
mkdir -p outputs/figures
python3 scripts/analysis/plot_eta_xq_valence_lowQ_overlay_two_beams.py \
  --preset five --out outputs/figures/Detectors.pdf
```

### Track A — `signals.pdf`

Five stages: **A1** generate editable parton-level Parquet → **A2** merge → **A3** alter (surgery **v2**,
required) → **A4** reinject + hadronize → observables CSV → **A5** plot. The generation (A1) is parallelized by
launching multiple worker processes; the alter (A3) and reinjection (A4) stages parallelize internally via
`--workers`. See [`AGENT_INSTRUCTIONS.md`](AGENT_INSTRUCTIONS.md) §A1–A5 for the exact commands.

Always pass **`--surgery-version v2`** to the alter step — v1 is the wrong model for this figure.

## Scale knobs

| Mode | Track B (`--n_events` per label) | Track A (accepted events, Q∈(4,5)) |
|------|----------------------------------|-------------------------------------|
| SMOKE (minutes) | `1000` | `2 × 500 = 1000` |
| Intermediate | `100000` | `12 × 40000 = 480k` |
| Paper-like (hours) | `100000` | `12 × 800000 = 9.6M` |

## Parallelism

- **Track B generation** is single-process (one PYTHIA instance, sequential over labels).
- **Track A** parallelizes three stages: A1 by launching N independent worker processes (disjoint `event_id`
  ranges), and A3/A4 via internal `multiprocessing` pools controlled by `--workers`. Effective parallelism is
  bounded by the number of shards, so choose `--events-per-shard` so that shards ≥ workers.

## Cleanup

`clean_tree.sh` reduces the working tree back to its known-good source files (useful if stray generated data ever
lands in the repo). It is **dry-run by default**:

```bash
./clean_tree.sh          # preview what would be removed
./clean_tree.sh --apply  # actually remove non-source files
```

It never deletes `.git/` or the script itself. Large data lives outside the repo, so normal runs never need it.

## Deliverables

After a successful run, confirm:

1. `outputs/figures/Detectors.pdf` (Track B)
2. `signals.pdf` under the Track A figures directory
3. Track A `run_summary.json` (under `altered_100k_parquet/`) reports `"surgery_version": "v2"`

## Citation

If you use this tooling, please cite the paper:

```bibtex
@article{Peng:2026src,
  title  = {Short-Range Correlations Between Partons in a Proton},
  author = {Peng, Jen-Chieh and Rajagopal, Krishna and Terry, John},
  eprint = {2606.17133},
  archivePrefix = {arXiv},
  primaryClass  = {hep-ph},
  note   = {MIT-CTP/6041},
  year   = {2026}
}
```

Event generation uses PYTHIA 8 (Sjöstrand et al., Comput. Phys. Commun. **191**, 159 (2015), [arXiv:1410.3012](https://arxiv.org/abs/1410.3012)).

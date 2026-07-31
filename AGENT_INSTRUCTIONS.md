# Agent instructions: generate both paper plots from scratch

**Working directory:** always the root of this tree (the directory that contains `scripts/` and `src/`).

```bash
cd "$(dirname "$0")"   # or: cd /path/to/Di-quark-pythia-standalone
```

This standalone package can produce:

| Plot | Track | Description |
|------|-------|-------------|
| **Detectors.pdf** | B | Five-beam leading-π lab η_h overlay with EIC detector bands |
| **signals.pdf** | A | Breit-frame φ_hJ: shared π⁺ + π⁻ at 0% / 5% / 10% altered fraction + R_{π⁻/π⁺} |

Do **not** confuse the tracks. They use different generators, data layouts, and plot scripts.

---

## Prerequisites

- Python 3.9+
- `numpy`, `pandas`, `pyarrow`, `matplotlib`, `tqdm`
- `pythia8` Python bindings (required for generation and Track A reinjection)

Optional environment:

```bash
export DIQUARK_DATA_ROOT="${HOME}/Data/Di-quark-pythia-standalone-nosync"   # Track B default data root
```

Long runs: wrap heavy steps in `caffeinate -i` on macOS so the machine does not sleep.

---

## Data location — do NOT melt the laptop (read before generating)

Generation writes **thousands of shard/Parquet files** (hundreds of MB to many GB). These must live **outside the repo and outside any cloud-synced or editor-indexed folder**. In the past, data landing in a synced location caused iCloud + Cursor indexing to thrash the disk/CPU simultaneously (the "laptop almost melted" incident).

**Rules:**

- **Keep all data under `~/Data/...`** (e.g. the default `~/Data/Di-quark-pythia-standalone-nosync/`). The `-nosync` suffix is intentional — it signals the folder must not be cloud-synced.
- **Never point `DIQUARK_DATA_ROOT`, `DIQUARK_OUTPUT_ROOT`, or `TRACK_A_ROOT` at anything under `~/Documents` or `~/Desktop`** (both are iCloud-synced on macOS). This repo itself lives under `~/Documents`, so **never let data be written inside the repo tree.**
- The only *enforced* protection is a soft `warnings.warn` in `src/diquark/paths.py` when the root resolves under `~/Documents`/`~/Desktop`. It does **not** stop the run — heed it.
- If you ever see generated `*.npy` / `*.parquet` files, `pythia_finalstate_raw/`, `chunks/`, `merged/`, or `outputs/` **inside the repo**, stop and move them out; they will be indexed/synced and can melt the machine.
- Confirm the target before long runs: `echo "$DIQUARK_DATA_ROOT" "$TRACK_A_ROOT"` and make sure both resolve under `~/Data`.

---

## Scale knobs

| Mode | Track B (`--n_events`) | Track A (accepted events in Q∈(4,5)) |
|------|------------------------|--------------------------------------|
| **SMOKE** (minutes) | `1000` per label | `2` workers × `500` = `1000` |
| **Paper-like** | `100000` per label | `12` workers × `800000` = `9.6M` (hours) |
| **Intermediate** | `100000` | `12` × `40000` = `480k` |

Below, **SMOKE** values are given first; replace with paper-like numbers for production.

---

# Track B — `Detectors.pdf` (η_h five-beam overlay)

## B1. Generate fully hadronized shards

```bash
python3 scripts/generation/generate_events_raw.py \
  --labels ETA_XQ_5x41,ETA_XQ_5x100,ETA_XQ_10x100,ETA_XQ_10x275,ETA_XQ_18x275 \
  --n_events 1000
```

Shards appear under:

`$DIQUARK_DATA_ROOT/pythia_finalstate_raw/ETA_XQ_*`

(default `~/Data/Di-quark-pythia-standalone-nosync/pythia_finalstate_raw/`).

## B2. Plot

```bash
mkdir -p outputs/figures
python3 scripts/analysis/plot_eta_xq_valence_lowQ_overlay_two_beams.py \
  --preset five \
  --out outputs/figures/Detectors.pdf
```

**Selection (inside plot script):** leading charged pion; **x > 0.05**; **4 GeV < Q < 5 GeV**; overlay five beams; shade Central / B0 / Forward η bands.

**Deliverable:** `outputs/figures/Detectors.pdf`


---

# Track A — `signals.pdf` (φ_hJ 0% / 5% / 10% signal)

Physics for the current paper-style figure:

- Beams **5 × 41 GeV**
- Generation acceptance: **x > 0.05**, **Q² ∈ (16, 25)** ⇒ **Q ∈ (4, 5) GeV**, struck **u**
- Surgery **v2**: s̄ fraction **0.01**, `x_us = 0.99 − 2 x_struck`, reject `x_struck > 0.495`
- Observables: reinject + hadronize; **π lab η ∈ (2.0, 3.5)**; **z_LC > 0.1** (hardcoded `ZLC_MIN` in `jet_hadron_observables_split_pi_pm.py`)
- Plot: shared unchanged π⁺; π⁻ at altered fractions **0, 0.05, 0.10**; ratio strip

Set a data root (example):

```bash
export TRACK_A_ROOT="${HOME}/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5"
mkdir -p "${TRACK_A_ROOT}/chunks" "${TRACK_A_ROOT}/merged"
```

## A1. Generate editable Parquet (parallel chunks)

Do **not** use `--kinematics-preset e5_ep41_x005_q2to4` alone for this bin — that preset forces **2 < Q < 4**. Use explicit Q² cuts for **4 < Q < 5**.

SMOKE (`2 × 500`):

```bash
N_WORKERS=2
N_CHUNK=500
BASE_SEED=12345

for w in $(seq 0 $((N_WORKERS - 1))); do
  OFF=$((w * N_CHUNK))
  S=$((BASE_SEED + w))
  python3 scripts/analysis/generate_dis_editable_source_parquet.py \
    --output-dir "${TRACK_A_ROOT}/chunks/chunk_${w}" \
    --n-accepted "${N_CHUNK}" \
    --event-id-offset "${OFF}" \
    --output-tag "w${w}" \
    --seed "${S}" \
    --beam-e 5 --beam-p 41 \
    --phase-space-q2-min 4 \
    --accept-x-min 0.05 \
    --accept-q2-min 16 \
    --accept-q2-max 25 \
    --events-per-shard 10000 \
    --kick-fraction 0 &
done
wait
```

Paper-like: set `N_WORKERS=12`, `N_CHUNK=800000` (or `40000` for 480k).

## A2. Merge chunks

```bash
CHUNKS=()
for w in $(seq 0 $((N_WORKERS - 1))); do
  CHUNKS+=("${TRACK_A_ROOT}/chunks/chunk_${w}")
done

python3 scripts/analysis/merge_editable_source_v1_chunks.py \
  --output-parent "${TRACK_A_ROOT}/merged" \
  --chunk-parents "${CHUNKS[@]}" \
  --events-per-shard 10000
```

## A3. Alter (surgery v2) — required

Default surgery is **v1**. Always pass **`--surgery-version v2`** (this tree only ships `split_dis_sample_diquark_kick_v2.py`).

```bash
caffeinate -i python3 scripts/analysis/transform_editable_all_altered_parquet.py \
  --editable-parent "${TRACK_A_ROOT}/merged" \
  --out-parent "${TRACK_A_ROOT}/merged" \
  --surgery-version v2 \
  --workers 12
```

Output: `${TRACK_A_ROOT}/merged/altered_100k_parquet/`

## A4. Observables → CSV

```bash
caffeinate -i python3 scripts/analysis/jet_hadron_observables_altered_parquet_filtered.py \
  --altered-root "${TRACK_A_ROOT}/merged/altered_100k_parquet" \
  --editable-parent "${TRACK_A_ROOT}/merged" \
  --out-csv "${TRACK_A_ROOT}/merged/jet_hadron_pi_pm_observables.csv" \
  --kinematics-preset e5_ep41_x005_q2to4 \
  --beam-e 5 --beam-p 41 \
  --dis-q-min 4 --dis-q-max 5 \
  --pion-eta-min 2.0 --pion-eta-max 3.5 \
  --workers 16
```

(`--kinematics-preset` here only sets beam/x defaults for reinjection kinematics; **`--dis-q-min/max`** enforce the Q∈(4,5) analysis window.)

## A5. Plot `signals.pdf`

```bash
FIG="${TRACK_A_ROOT}/merged/figures_pi_pm_observables"
CSV="${TRACK_A_ROOT}/merged/jet_hadron_pi_pm_observables.csv"
CAPTION=$'$x > 0.05$\n$4~\\mathrm{GeV} < Q < 5~\\mathrm{GeV}$\n$2.0 < \\eta_h^{\\mathrm{lab}} < 3.5$\n$z_{LC} > 0.1$\n$E_e \\times E_p = 5 \\times 41~\\mathrm{GeV}^2$'

mkdir -p "${FIG}"

python3 scripts/analysis/plot_jet_hadron_pi_pm_observables.py \
  --csv "${CSV}" \
  --figure-dir "${FIG}" \
  --layout combined \
  --only-ok-rows \
  --combined-only angle \
  --compare-pooled-altered-fractions 0,0.05,0.10 \
  --pooled-balance-seed 42 \
  --combined-filename-suffix _signals_v2_sbar01_zLC01 \
  --no-title \
  --caption "${CAPTION}"

# Convenience copy / rename for the paper figure name
cp "${FIG}/jet_hadron_transverse_angle_target_leading_pion_Breit_combined_pi_pm_signals_v2_sbar01_zLC01.pdf" \
   "${FIG}/signals.pdf"
mkdir -p outputs/figures
cp "${FIG}/signals.pdf" outputs/figures/signals.pdf
```

**What the plot does:**

1. π⁺ (blue) = all `ok` **unchanged** rows (shared reference)
2. π⁻ curves = pooled mixtures with altered fraction **0 / 0.05 / 0.10**
3. Bottom strip = R_{π⁻/π⁺} for each mixture

### Optional: 1×3 talk triptych (same CSV)

```bash
python3 scripts/analysis/plot_jet_hadron_pi_pm_observables.py \
  --csv "${CSV}" \
  --figure-dir "${FIG}" \
  --layout triptych \
  --only-ok-rows \
  --pooled-balance-seed 42 \
  --combined-filename-suffix _signals_v2_sbar01_zLC01 \
  --triptych-standalone-panels \
  --caption "${CAPTION}"
```

Combined triptych keeps grey / purple / red π⁻ across panels; standalone single-panel talk PDFs force π⁻ **red**.


---

## Agent checklist (deliverables)

After a successful run, confirm these exist:

1. `outputs/figures/Detectors.pdf` (or the `--out` path you chose in B2)
2. `outputs/figures/signals.pdf` (or `${FIG}/signals.pdf` from A5)
3. Track A `run_summary.json` under `altered_100k_parquet/` shows `"surgery_version": "v2"`

## What NOT to do

- Do not write data inside the repo or under `~/Documents` / `~/Desktop` (iCloud-synced). Keep all data under `~/Data/...` — see "Data location — do NOT melt the laptop" above.
- Do not run Track A alteration without `--surgery-version v2`.
- Do not use `--kinematics-preset e5_ep41_x005_q2to4` for **Track A generation** of the Q∈(4,5) sample (wrong Q² window).
- Do not omit `--pion-eta-min 2.0 --pion-eta-max 3.5` on observables if matching the paper caption.
- Do not mix Track B NumPy shards with Track A Parquet paths.
- Do not regenerate if the user only asked to replot — plot scripts alone suffice when CSV / shards already exist.

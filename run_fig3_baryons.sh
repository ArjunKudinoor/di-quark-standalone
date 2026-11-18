#!/usr/bin/env bash
set -euo pipefail

# Fig. 3 p/n follow-up. Same A1--A5 workflow as run_fig3_pi0.sh.
# Paper-scale defaults: 12 x 800000 accepted events, original seeds/cards.
# Existing original/pi0 A1--A3 products are preferred; never overwritten.
# Usage: ./run_fig3_baryons.sh
# Resume analysis: ./run_fig3_baryons.sh --start A4
# Plot completed baryon observables only: ./run_fig3_baryons.sh --start A5

N_CHUNK=800000
N_CHUNKS=12
MAX_GEN_JOBS=4
WORKERS=4
START_STAGE=A1
BASE_SEED=12345
SOURCE_ROOT=""

# Preserve the pi0 runner's environment conventions.
if [[ -z "${VIRTUAL_ENV:-}" && -x .venv/bin/python3 ]]; then
    export PATH="${PWD}/.venv/bin:${PATH}"
fi
if ! python3 -c 'import pythia8' 2>/dev/null; then
    LOCAL_PYTHIA="${PWD}/../software/pythia8315/lib"
    if [[ -f "${LOCAL_PYTHIA}/pythia8.so" ]]; then
        export PYTHONPATH="${LOCAL_PYTHIA}${PYTHONPATH:+:${PYTHONPATH}}"
    fi
fi
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/diquark-fig3-baryons-mpl}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --events-per-chunk|--chunks|--gen-jobs|--workers|--start|--source-root)
            if [[ $# -lt 2 ]]; then
                echo "ERROR: missing value for $1"; exit 1
            fi
            case "$1" in
                --events-per-chunk) N_CHUNK="$2" ;;
                --chunks) N_CHUNKS="$2" ;;
                --gen-jobs) MAX_GEN_JOBS="$2" ;;
                --workers) WORKERS="$2" ;;
                --start) START_STAGE="$2" ;;
                --source-root) SOURCE_ROOT="$2" ;;
            esac
            shift 2 ;;
        -h|--help)
            echo "Usage: ./run_fig3_baryons.sh [options]"
            echo "  --events-per-chunk N  Accepted events per chunk (default 800000)"
            echo "  --chunks N            Chunks (default 12)"
            echo "  --gen-jobs N          Concurrent A1 generators (default 4)"
            echo "  --workers N           A3/A4 workers (default 4)"
            echo "  --start A1|A2|A3|A4|A5 (default A1)"
            echo "  --source-root PATH    Existing original/pi0 root containing merged/"
            exit 0 ;;
        *) echo "ERROR: unknown argument $1"; exit 1 ;;
    esac
done

for value in "${N_CHUNK}" "${N_CHUNKS}" "${MAX_GEN_JOBS}" "${WORKERS}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ || ${#value} -gt 7 ]]; then
        echo "ERROR: counts must be positive integers of at most seven digits."
        exit 1
    fi
done
# Decimal parsing avoids Bash's octal interpretation of leading zeroes.
N_CHUNK=$((10#${N_CHUNK}))
N_CHUNKS=$((10#${N_CHUNKS}))
MAX_GEN_JOBS=$((10#${MAX_GEN_JOBS}))
WORKERS=$((10#${WORKERS}))
if (( N_CHUNK <= 0 || N_CHUNKS <= 0 || MAX_GEN_JOBS <= 0 || WORKERS <= 0 )); then
    echo "ERROR: counts must be positive."; exit 1
fi
TOTAL_EVENTS=$((N_CHUNK * N_CHUNKS))
if (( TOTAL_EVENTS > 9600000 )); then
    echo "ERROR: maximum size is 9,600,000 accepted events."; exit 1
fi
stage_number() {
    case "$1" in
        A1) echo 1 ;; A2) echo 2 ;; A3) echo 3 ;; A4) echo 4 ;; A5) echo 5 ;;
        *) echo "ERROR: invalid stage $1" >&2; return 1 ;;
    esac
}
START_NUM=$(stage_number "${START_STAGE}")
should_run() { (( $(stage_number "$1") >= START_NUM )); }

DATA_BASE="${HOME}/Data/dis_q_bins_e5_ep41_standalone"
TRACK_A_ROOT="${DATA_BASE}/Q2_16_25__Q_4_5_${TOTAL_EVENTS}_baryons"
CHUNKS_ROOT="${TRACK_A_ROOT}/chunks"
MERGED_ROOT="${TRACK_A_ROOT}/merged"
CSV="${MERGED_ROOT}/jet_hadron_baryon_observables.csv"
FIG="outputs/figures/fig3_baryons_${TOTAL_EVENTS}"
SUMMARY="outputs/analysis/fig3_baryons_${TOTAL_EVENTS}"

if [[ -n "${SOURCE_ROOT}" ]]; then
    if [[ ! -d "${SOURCE_ROOT}/merged/editable_source_v1" ]]; then
        echo "ERROR: --source-root has no merged/editable_source_v1"; exit 1
    fi
    SOURCE_ROOT=$(cd "${SOURCE_ROOT}" && pwd -P)
else
    # The existing paper sample currently resides in the pi0-specific root.
    for candidate in \
        "${DATA_BASE}/Q2_16_25__Q_4_5_${TOTAL_EVENTS}_pi0" \
        "${DATA_BASE}/Q2_16_25__Q_4_5_${TOTAL_EVENTS}" \
        "${TRACK_A_ROOT}"; do
        if [[ -d "${candidate}/merged/editable_source_v1" ]]; then
            SOURCE_ROOT="${candidate}"
            break
        fi
    done
    if [[ -z "${SOURCE_ROOT}" && "${TOTAL_EVENTS}" == 480000 && -d "${DATA_BASE}/Q2_16_25__Q_4_5_intermediate/merged/editable_source_v1" ]]; then
        SOURCE_ROOT="${DATA_BASE}/Q2_16_25__Q_4_5_intermediate"
    fi
    SOURCE_ROOT="${SOURCE_ROOT:-${TRACK_A_ROOT}}"
fi

echo "============================================================"
echo "                 FIG. 3 BARYON PIPELINE"
echo "============================================================"
echo "Events/chunk: ${N_CHUNK}; chunks: ${N_CHUNKS}; accepted: ${TOTAL_EVENTS}"
echo "A1 jobs: ${MAX_GEN_JOBS}; A3/A4 workers: ${WORKERS}; starting: ${START_STAGE}"
echo "Input: ${SOURCE_ROOT}"
echo "New data: ${TRACK_A_ROOT}"
echo "Figures: ${FIG}"

if [[ "${START_STAGE}" == A5 ]]; then
    if [[ ! -f "${CSV}" || ! -f "${SUMMARY}/observables_summary.json" ]]; then
        echo "ERROR: A5 needs a completed baryon CSV and its A4 summary. No events generated."
        exit 1
    fi
else
    # Always nonempty: safe with macOS Bash 3.2 and set -u.
    PREFLIGHT=(python3 scripts/analysis/validate_fig3_baryons_inputs.py
        --source-parent "${SOURCE_ROOT}/merged" --expected-events "${TOTAL_EVENTS}" --workers "${WORKERS}")
    if (( START_NUM >= 3 )); then PREFLIGHT+=(--require-editable); fi
    if (( START_NUM >= 4 )); then PREFLIGHT+=(--require-altered); fi
    "${PREFLIGHT[@]}"
    mkdir -p "${CHUNKS_ROOT}" "${MERGED_ROOT}" "${SUMMARY}"
    "${PREFLIGHT[@]}" --manifest "${SUMMARY}/configuration.json"
    # Input directory links only. A1--A3 never write through these links.
    for dataset in editable_source_v1 altered_100k_parquet; do
        if [[ -d "${SOURCE_ROOT}/merged/${dataset}" && ! -e "${MERGED_ROOT}/${dataset}" && ! -L "${MERGED_ROOT}/${dataset}" ]]; then
            ln -s "${SOURCE_ROOT}/merged/${dataset}" "${MERGED_ROOT}/${dataset}"
        fi
    done
fi

# A1 -- original editable DIS generator, only if there is no reusable source.
if should_run A1; then
    if [[ -d "${MERGED_ROOT}/editable_source_v1" ]]; then
        echo "A1: reusing existing accepted DIS source."
    else
        echo "========== A1: GENERATING ${TOTAL_EVENTS} EVENTS =========="
        PIDS=()
        for w in $(seq 0 $((N_CHUNKS - 1))); do
            if [[ -e "${CHUNKS_ROOT}/chunk_${w}" ]]; then
                echo "ERROR: existing chunk_${w}; resume with --start instead of overwriting."; exit 1
            fi
            while (( $(jobs -rp | wc -l | tr -d ' ') >= MAX_GEN_JOBS )); do sleep 1; done
            OFF=$((w * N_CHUNK))
            S=$((BASE_SEED + w))
            echo "Starting chunk ${w}: accepted=${N_CHUNK}, seed=${S}, offset=${OFF}"
            python3 scripts/analysis/generate_dis_editable_source_parquet.py \
                --output-dir "${CHUNKS_ROOT}/chunk_${w}" --n-accepted "${N_CHUNK}" \
                --event-id-offset "${OFF}" --output-tag "w${w}" --seed "${S}" \
                --beam-e 5 --beam-p 41 --phase-space-q2-min 4 \
                --accept-x-min 0.05 --accept-q2-min 16 --accept-q2-max 25 \
                --events-per-shard 10000 --kick-fraction 0 &
            PIDS+=("$!")
        done
        for pid in "${PIDS[@]}"; do wait "${pid}"; done
        echo "A1 complete."
    fi
else echo "Skipping A1."; fi

# A2 -- original chunk merger into the new baryon root.
if should_run A2; then
    if [[ -d "${MERGED_ROOT}/editable_source_v1" ]]; then
        echo "A2: reusing existing merged source."
    else
        echo "========== A2: MERGING CHUNKS =========="
        if [[ -e "${MERGED_ROOT}/editable_source_v1" || -L "${MERGED_ROOT}/editable_source_v1" ]]; then
            echo "ERROR: existing editable destination; refusing to overwrite."; exit 1
        fi
        CHUNKS=()
        for w in $(seq 0 $((N_CHUNKS - 1))); do CHUNKS+=("${CHUNKS_ROOT}/chunk_${w}"); done
        python3 scripts/analysis/merge_editable_source_v1_chunks.py \
            --output-parent "${MERGED_ROOT}" --chunk-parents "${CHUNKS[@]}" --events-per-shard 10000
        echo "A2 complete."
    fi
else echo "Skipping A2."; fi

# A3 -- original v2 surgery; an existing original/pi0 alteration is reused.
if should_run A3; then
    if [[ -d "${MERGED_ROOT}/altered_100k_parquet" ]]; then
        echo "A3: reusing existing v2 alteration."
    else
        echo "========== A3: DIQUARK SURGERY =========="
        if [[ -e "${MERGED_ROOT}/altered_100k_parquet" || -L "${MERGED_ROOT}/altered_100k_parquet" ]]; then
            echo "ERROR: existing altered destination; refusing to overwrite."; exit 1
        fi
        caffeinate -i python3 scripts/analysis/transform_editable_all_altered_parquet.py \
            --editable-parent "${MERGED_ROOT}" --out-parent "${MERGED_ROOT}" \
            --surgery-version v2 --seed 42 --workers "${WORKERS}"
        echo "A3 complete."
    fi
else echo "Skipping A3."; fi

# A4 -- new species selection, same rehadronization and physics settings.
if should_run A4; then
    if [[ -f "${CSV}" && -f "${SUMMARY}/observables_summary.json" ]]; then
        echo "A4: preserving existing completed baryon observables."
    elif [[ -e "${CSV}" || -e "${SUMMARY}/observables_summary.json" ]]; then
        echo "ERROR: incomplete/existing A4 product; refusing to overwrite."; exit 1
    else
        echo "========== A4: CALCULATING BARYON OBSERVABLES =========="
        caffeinate -i python3 scripts/analysis/jet_hadron_observables_altered_parquet_filtered_baryons.py \
            --altered-root "${MERGED_ROOT}/altered_100k_parquet" --editable-parent "${MERGED_ROOT}" \
            --out-csv "${CSV}" --summary-json "${SUMMARY}/observables_summary.json" \
            --beam-e 5 --beam-p 41 --dis-x-floor 0.05 --dis-q-min 4 --dis-q-max 5 \
            --baryon-eta-min 2.0 --baryon-eta-max 3.5 --workers "${WORKERS}"
        echo "A4 complete."
    fi
else echo "Skipping A4."; fi

# A5 -- same-pool n/p plus species-specific 5%/0% and 10%/0% comparisons.
if should_run A5; then
    echo "========== A5: PLOTTING BARYON FIG. 3 =========="
    CAPTION=$'$x > 0.05$\n$4~\\mathrm{GeV} < Q < 5~\\mathrm{GeV}$\n$2.0 < \\eta_h^{\\mathrm{lab}} < 3.5$\n$z_{LC} > 0.1$\n$E_e \\times E_p = 5 \\times 41~\\mathrm{GeV}^2$'
    python3 scripts/analysis/plot_jet_hadron_baryon_observables.py \
        --csv "${CSV}" --observables-summary "${SUMMARY}/observables_summary.json" \
        --figure-dir "${FIG}" --summary-dir "${SUMMARY}/plots" \
        --compare-pooled-altered-fractions 0,0.05,0.10 --pooled-balance-seed 42 --caption "${CAPTION}"
    echo "A5 complete."
fi

echo "============================================================"
echo "DONE"
echo "Dataset: ${TRACK_A_ROOT}"
echo "Figures: ${FIG}/fig3_baryons_src{0,5,10}.pdf"
echo "SRC comparisons: ${FIG}/fig3_{neutron,proton}_src_comparison.pdf"

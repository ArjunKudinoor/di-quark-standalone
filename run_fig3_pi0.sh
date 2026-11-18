#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Fig. 3 reproduction pipeline including neutral pions
#
# Defaults = smoke test:
#   500 events/chunk × 2 chunks = 1,000 accepted events
#   2 simultaneous A1 generators
#   2 workers for A3/A4
#
# Usage:
#
#   ./run_fig3_pi0.sh
#
# Add the pi+ comparison from an existing paper-scale observable CSV (A5 only):
#   ./run_fig3_pi0.sh --events-per-chunk 800000 --chunks 12 --piplus-only
#
#   ./run_fig3_pi0.sh --events-per-chunk 40000 \
#                 --chunks 12 \
#                 --gen-jobs 4 \
#                 --workers 4
#
# Resume from a particular stage:
#
#   ./run_fig3_pi0.sh --events-per-chunk 40000 \
#                 --chunks 12 \
#                 --gen-jobs 4 \
#                 --workers 4 \
#                 --start A3
#
# Stages: A1, A2, A3, A4, A5
# ============================================================


# ------------------------------------------------------------
# Defaults
# ------------------------------------------------------------

N_CHUNK=500
N_CHUNKS=2
MAX_GEN_JOBS=2
WORKERS=2
START_STAGE="A1"
PIPLUS_ONLY=0

BASE_SEED=12345

# Use this checkout's existing dependencies when no environment is active.
# An explicitly activated environment and PYTHONPATH continue to take precedence.
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
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/diquark-fig3-pi0-mpl}"



# ------------------------------------------------------------
# Parse command-line arguments
# ------------------------------------------------------------

while [[ $# -gt 0 ]]; do
    case "$1" in

        --events-per-chunk)
            N_CHUNK="$2"
            shift 2
            ;;

        --chunks)
            N_CHUNKS="$2"
            shift 2
            ;;

        --gen-jobs)
            MAX_GEN_JOBS="$2"
            shift 2
            ;;

        --workers)
            WORKERS="$2"
            shift 2
            ;;

        --start)
            START_STAGE="$2"
            shift 2
            ;;

        --piplus-only)
            PIPLUS_ONLY=1
            shift
            ;;

        -h|--help)
            echo "Usage: ./run_fig3_pi0.sh [options]"
            echo
            echo "Options:"
            echo "  --events-per-chunk N   Accepted events per chunk (default: 500)"
            echo "  --chunks N             Number of chunks (default: 2)"
            echo "  --gen-jobs N           Maximum simultaneous A1 generators (default: 2)"
            echo "  --workers N            Workers for A3/A4 (default: 2)"
            echo "  --start STAGE          Start at A1, A2, A3, A4, or A5"
            echo "  --piplus-only          Make only the pi+ SRC comparison from the existing CSV (A5 only)"
            echo "  -h, --help             Show this help"
            exit 0
            ;;

        *)
            echo "ERROR: Unknown argument: $1"
            exit 1
            ;;
    esac
done


# ------------------------------------------------------------
# Validate arguments
# ------------------------------------------------------------

case "${START_STAGE}" in
    A1|A2|A3|A4|A5)
        ;;
    *)
        echo "ERROR: --start must be A1, A2, A3, A4, or A5"
        exit 1
        ;;
esac

if (( PIPLUS_ONLY )); then
    START_STAGE="A5"
fi

for value in "${N_CHUNK}" "${N_CHUNKS}" "${MAX_GEN_JOBS}" "${WORKERS}"; do
    if [[ ! "${value}" =~ ^[0-9]+$ || ${#value} -gt 7 ]]; then
        echo "ERROR: counts must be positive integers of at most seven digits."
        exit 1
    fi
done

if (( N_CHUNK <= 0 || N_CHUNKS <= 0 || MAX_GEN_JOBS <= 0 || WORKERS <= 0 )); then
    echo "ERROR: event counts and worker counts must be positive integers."
    exit 1
fi


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TOTAL_EVENTS=$((N_CHUNK * N_CHUNKS))
if (( TOTAL_EVENTS > 9600000 )); then
    echo "ERROR: this follow-up supports at most 9,600,000 accepted DIS events."
    exit 1
fi

TRACK_A_ROOT="${HOME}/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_${TOTAL_EVENTS}_pi0"
# Reuse A1–A3 products only; never write through these source directories.
SOURCE_ROOT="${HOME}/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_${TOTAL_EVENTS}"
if (( TOTAL_EVENTS == 480000 )) && [[ ! -d "${SOURCE_ROOT}/merged/editable_source_v1" ]]; then
    SOURCE_ROOT="${HOME}/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_intermediate"
fi

CHUNKS_ROOT="${TRACK_A_ROOT}/chunks"
MERGED_ROOT="${TRACK_A_ROOT}/merged"

CSV="${MERGED_ROOT}/jet_hadron_pi_pm_pi0_observables.csv"
FIG="outputs/figures/fig3_pi0_${TOTAL_EVENTS}"
if (( TOTAL_EVENTS == 480000 )); then
    FIG="outputs/figures"
fi
SUMMARY="outputs/analysis/fig3_pi0_${TOTAL_EVENTS}"
# Keep the previous shared-0%-pi+ numerical products intact.
PLOT_SUMMARY="${SUMMARY}/matched_piplus"
if (( PIPLUS_ONLY )); then
    PLOT_SUMMARY="${SUMMARY}"
fi

# Read-only preflight before creating output paths.
if (( PIPLUS_ONLY )) || [[ "${START_STAGE}" == "A5" ]]; then
    if [[ ! -f "${CSV}" ]]; then
        echo "ERROR: plotting requires the existing observable CSV: ${CSV}"
        echo "No events will be generated. Use --start A4 to extract observables from existing partons first."
        exit 1
    fi
    mkdir -p "${FIG}" "${PLOT_SUMMARY}"
else
    python3 scripts/analysis/validate_fig3_pi0_inputs.py \
        --source-parent "${SOURCE_ROOT}/merged" --expected-events "${TOTAL_EVENTS}"
    mkdir -p "${CHUNKS_ROOT}" "${MERGED_ROOT}" "${FIG}" "${SUMMARY}"
    python3 scripts/analysis/validate_fig3_pi0_inputs.py \
        --source-parent "${SOURCE_ROOT}/merged" --expected-events "${TOTAL_EVENTS}" \
        --manifest "${SUMMARY}/configuration.json" --workers "${WORKERS}"

    # Prepare read-only input links even when resuming directly at A4/A5.
    for dataset in editable_source_v1 altered_100k_parquet; do
        if [[ -d "${SOURCE_ROOT}/merged/${dataset}" && ! -e "${MERGED_ROOT}/${dataset}" ]]; then
            ln -s "${SOURCE_ROOT}/merged/${dataset}" "${MERGED_ROOT}/${dataset}"
        fi
    done
fi

# ------------------------------------------------------------
# Helper: determine whether a stage should run
# ------------------------------------------------------------

stage_number() {
    case "$1" in
        A1) echo 1 ;;
        A2) echo 2 ;;
        A3) echo 3 ;;
        A4) echo 4 ;;
        A5) echo 5 ;;
    esac
}

START_NUM=$(stage_number "${START_STAGE}")

should_run() {
    local stage_num
    stage_num=$(stage_number "$1")
    (( stage_num >= START_NUM ))
}


# ------------------------------------------------------------
# Print configuration
# ------------------------------------------------------------

echo
echo "============================================================"
echo "                 FIG. 3 PI0 PIPELINE"
echo "============================================================"
echo
echo "Events/chunk       : ${N_CHUNK}"
echo "Chunks             : ${N_CHUNKS}"
echo "Total events       : ${TOTAL_EVENTS}"
echo "Max A1 generators  : ${MAX_GEN_JOBS}"
echo "A3/A4 workers      : ${WORKERS}"
echo "Starting at        : ${START_STAGE}"
echo
echo "Data:"
echo "  ${TRACK_A_ROOT}"
echo
echo "============================================================"


# ============================================================
# A1 — Generate editable DIS events
# ============================================================

if should_run A1 && [[ -d "${SOURCE_ROOT}/merged/editable_source_v1" ]]; then
    echo "A1: reusing existing ${TOTAL_EVENTS}-event source at ${SOURCE_ROOT}."
elif should_run A1; then

    echo
    if [[ -d "${MERGED_ROOT}/editable_source_v1" ]]; then
        echo "A1: merged follow-up source exists; use --start A3 or later."
        exit 1
    fi
    echo "========== A1: GENERATING ${TOTAL_EVENTS} EVENTS =========="

    PIDS=()
    for w in $(seq 0 $((N_CHUNKS - 1))); do

        # Wait until fewer than MAX_GEN_JOBS processes are running.
        while (( $(jobs -rp | wc -l | tr -d ' ') >= MAX_GEN_JOBS )); do
            sleep 1
        done

        OFF=$((w * N_CHUNK))
        S=$((BASE_SEED + w))

        echo
        echo "Starting chunk ${w}/${N_CHUNKS}"
        echo "  events = ${N_CHUNK}"
        echo "  seed   = ${S}"
        echo "  offset = ${OFF}"

        if [[ -e "${CHUNKS_ROOT}/chunk_${w}" ]]; then
            echo "ERROR: existing chunk_${w}; use --start to resume without regenerating."
            exit 1
        fi
        python3 scripts/analysis/generate_dis_editable_source_parquet.py \
            --output-dir "${CHUNKS_ROOT}/chunk_${w}" \
            --n-accepted "${N_CHUNK}" \
            --event-id-offset "${OFF}" \
            --output-tag "w${w}" \
            --seed "${S}" \
            --beam-e 5 \
            --beam-p 41 \
            --phase-space-q2-min 4 \
            --accept-x-min 0.05 \
            --accept-q2-min 16 \
            --accept-q2-max 25 \
            --events-per-shard 10000 \
            --kick-fraction 0 &
        PIDS+=("$!")

    done

    echo
    echo "Waiting for remaining A1 jobs..."

    for pid in "${PIDS[@]}"; do
        wait "${pid}"
    done

    echo
    echo "A1 complete."

else

    echo
    echo "Skipping A1."

fi


# ============================================================
# A2 — Merge chunks
# ============================================================

if should_run A2 && [[ -d "${SOURCE_ROOT}/merged/editable_source_v1" ]]; then
    echo "A2: reusing merged editable source (read-only directory link)."
    if [[ ! -e "${MERGED_ROOT}/editable_source_v1" ]]; then
        ln -s "${SOURCE_ROOT}/merged/editable_source_v1" "${MERGED_ROOT}/editable_source_v1"
    fi
elif should_run A2; then

    echo
    echo "========== A2: MERGING CHUNKS =========="

    CHUNKS=()

    for w in $(seq 0 $((N_CHUNKS - 1))); do
        CHUNKS+=("${CHUNKS_ROOT}/chunk_${w}")
    done

    if [[ -e "${MERGED_ROOT}/editable_source_v1" ]]; then
        echo "ERROR: existing merged source; use --start A3 or later."
        exit 1
    fi
    python3 scripts/analysis/merge_editable_source_v1_chunks.py \
        --output-parent "${MERGED_ROOT}" \
        --chunk-parents "${CHUNKS[@]}" \
        --events-per-shard 10000

    echo
    echo "A2 complete."

else

    echo
    echo "Skipping A2."

fi


# ============================================================
# A3 — Diquark surgery
# ============================================================

if should_run A3 && [[ -d "${SOURCE_ROOT}/merged/altered_100k_parquet" ]]; then
    echo "A3: reusing existing v2 alteration (read-only directory link)."
    if [[ ! -e "${MERGED_ROOT}/altered_100k_parquet" ]]; then
        ln -s "${SOURCE_ROOT}/merged/altered_100k_parquet" "${MERGED_ROOT}/altered_100k_parquet"
    fi
elif should_run A3 && [[ -d "${MERGED_ROOT}/altered_100k_parquet" ]]; then
    echo "A3: existing follow-up alteration is preserved."
elif should_run A3; then

    echo
    echo "========== A3: DIQUARK SURGERY =========="

    caffeinate -i python3 scripts/analysis/transform_editable_all_altered_parquet.py \
        --editable-parent "${MERGED_ROOT}" \
        --out-parent "${MERGED_ROOT}" \
        --surgery-version v2 \
        --workers "${WORKERS}"

    echo
    echo "A3 complete."

else

    echo
    echo "Skipping A3."

fi


# ============================================================
# A4 — Reinjection / hadronization / observables
# ============================================================

if should_run A4 && [[ -f "${CSV}" ]]; then
    echo "A4: preserving existing follow-up observable CSV; continuing to plotting."
elif should_run A4; then

    echo
    echo "========== A4: CALCULATING OBSERVABLES =========="

    caffeinate -i python3 scripts/analysis/jet_hadron_observables_altered_parquet_filtered_pi0.py \
        --altered-root "${MERGED_ROOT}/altered_100k_parquet" \
        --editable-parent "${MERGED_ROOT}" \
        --out-csv "${CSV}" \
        --summary-json "${SUMMARY}/observables_summary.json" \
        --kinematics-preset e5_ep41_x005_q2to4 \
        --beam-e 5 \
        --beam-p 41 \
        --dis-q-min 4 \
        --dis-q-max 5 \
        --pion-eta-min 2.0 \
        --pion-eta-max 3.5 \
        --workers "${WORKERS}"

    echo
    echo "A4 complete."

else

    echo
    echo "Skipping A4."

fi


# ============================================================
# A5 — Plot Fig. 3
# ============================================================

if should_run A5; then

    echo
    echo "========== A5: PLOTTING FIG. 3 =========="

    mkdir -p "${FIG}"

    CAPTION=$'$x > 0.05$\n$4~\\mathrm{GeV} < Q < 5~\\mathrm{GeV}$\n$2.0 < \\eta_h^{\\mathrm{lab}} < 3.5$\n$z_{LC} > 0.1$\n$E_e \\times E_p = 5 \\times 41~\\mathrm{GeV}^2$'

    # Bash 3.2 treats an empty array as unset under `set -u`.
    # Include the command itself so the array is always nonempty.
    PLOT_COMMAND=(python3 scripts/analysis/plot_jet_hadron_pi_pm_pi0_observables.py)
    if (( PIPLUS_ONLY )); then
        PLOT_COMMAND+=(--piplus-only)
    fi
    "${PLOT_COMMAND[@]}" \
        --csv "${CSV}" \
        --figure-dir "${FIG}" \
        --summary-dir "${PLOT_SUMMARY}" \
        --layout combined \
        --only-ok-rows \
        --combined-only angle \
        --compare-pooled-altered-fractions 0,0.05,0.10 \
        --pooled-balance-seed 42 \
        --combined-filename-suffix _signals_v2_sbar01_zLC01 \
        --no-title \
        --caption "${CAPTION}"

    FINAL_PDF="${FIG}/fig3_pi0_src{0,5,10}.pdf"
    if (( PIPLUS_ONLY )); then
        FINAL_PDF="${FIG}/fig3_piplus_src_comparison.pdf"
    else
        echo "Additional pi+ comparison: ${FIG}/fig3_piplus_src_comparison.pdf"
    fi

    echo
    echo "A5 complete."

else

    echo
    echo "Skipping A5."

fi


# ============================================================
# Done
# ============================================================

echo
echo "============================================================"
echo "DONE"
echo "============================================================"
echo
echo "Dataset:"
echo "  ${TRACK_A_ROOT}"
echo
echo "Figure:"
echo "  ${FINAL_PDF}"
echo

# if [[ -f "${FINAL_PDF}" ]]; then
#     open "${FINAL_PDF}"
# fi

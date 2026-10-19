#!/usr/bin/env bash

set -euo pipefail

# ============================================================
# Fig. 3 reproduction pipeline
#
# Defaults = smoke test:
#   500 events/chunk × 2 chunks = 1,000 accepted events
#   2 simultaneous A1 generators
#   2 workers for A3/A4
#
# Usage:
#
#   ./run_fig3.sh
#
#   ./run_fig3.sh --events-per-chunk 40000 \
#                 --chunks 12 \
#                 --gen-jobs 4 \
#                 --workers 4
#
# Resume from a particular stage:
#
#   ./run_fig3.sh --events-per-chunk 40000 \
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

BASE_SEED=12345


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

        -h|--help)
            echo "Usage: ./run_fig3.sh [options]"
            echo
            echo "Options:"
            echo "  --events-per-chunk N   Accepted events per chunk (default: 500)"
            echo "  --chunks N             Number of chunks (default: 2)"
            echo "  --gen-jobs N           Maximum simultaneous A1 generators (default: 2)"
            echo "  --workers N            Workers for A3/A4 (default: 2)"
            echo "  --start STAGE          Start at A1, A2, A3, A4, or A5"
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

if (( N_CHUNK <= 0 || N_CHUNKS <= 0 || MAX_GEN_JOBS <= 0 || WORKERS <= 0 )); then
    echo "ERROR: event counts and worker counts must be positive integers."
    exit 1
fi


# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

TOTAL_EVENTS=$((N_CHUNK * N_CHUNKS))

TRACK_A_ROOT="${HOME}/Data/dis_q_bins_e5_ep41_standalone/Q2_16_25__Q_4_5_${TOTAL_EVENTS}"

CHUNKS_ROOT="${TRACK_A_ROOT}/chunks"
MERGED_ROOT="${TRACK_A_ROOT}/merged"

CSV="${MERGED_ROOT}/jet_hadron_pi_pm_observables.csv"
FIG="${MERGED_ROOT}/figures_pi_pm_observables"

mkdir -p "${CHUNKS_ROOT}" "${MERGED_ROOT}" outputs/figures


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
echo "                 FIG. 3 PIPELINE"
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

if should_run A1; then

    echo
    echo "========== A1: GENERATING ${TOTAL_EVENTS} EVENTS =========="

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

    done

    echo
    echo "Waiting for remaining A1 jobs..."

    wait

    echo
    echo "A1 complete."

else

    echo
    echo "Skipping A1."

fi


# ============================================================
# A2 — Merge chunks
# ============================================================

if should_run A2; then

    echo
    echo "========== A2: MERGING CHUNKS =========="

    CHUNKS=()

    for w in $(seq 0 $((N_CHUNKS - 1))); do
        CHUNKS+=("${CHUNKS_ROOT}/chunk_${w}")
    done

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

if should_run A3; then

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

if should_run A4; then

    echo
    echo "========== A4: CALCULATING OBSERVABLES =========="

    caffeinate -i python3 scripts/analysis/jet_hadron_observables_altered_parquet_filtered.py \
        --altered-root "${MERGED_ROOT}/altered_100k_parquet" \
        --editable-parent "${MERGED_ROOT}" \
        --out-csv "${CSV}" \
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

    SOURCE_PDF="${FIG}/jet_hadron_transverse_angle_target_leading_pion_Breit_combined_pi_pm_signals_v2_sbar01_zLC01.pdf"

    FINAL_PDF="outputs/figures/signals_${TOTAL_EVENTS}.pdf"
    cp "${SOURCE_PDF}" "${FINAL_PDF}"

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

#!/usr/bin/env bash
#
# clean_tree.sh — remove everything in the repo tree that is not part of the
# known-good source set (the "manifest" below).
#
# Purpose: bulk simulation output (*.npy, *.parquet, pythia_finalstate_raw/,
# chunks/, merged/, outputs/, ...) must never live inside this repo (it sits
# under ~/Documents, which is iCloud-synced + Cursor-indexed). If any stray
# generated files land here, this script wipes them back to a clean source tree.
#
# SAFETY:
#   - Dry-run by default: prints what WOULD be deleted, changes nothing.
#   - Pass --apply to actually delete.
#   - Never deletes the manifest files, this script itself, or the .git/ dir.
#
# Usage:
#   ./clean_tree.sh            # dry run (safe preview)
#   ./clean_tree.sh --apply    # actually delete non-manifest files
#
set -euo pipefail

# Repo root = directory containing this script.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SELF="$(basename "${BASH_SOURCE[0]}")"

# --- Known-good file manifest (relative to ROOT) ---------------------------
# Keep this list in sync with the intended source tree.
MANIFEST=(
  "AGENT_INSTRUCTIONS.md"
  "scripts/analysis/analyze_jet_hadron_transverse_observables.py"
  "scripts/analysis/editable_source_parquet.py"
  "scripts/analysis/generate_dis_editable_source_parquet.py"
  "scripts/analysis/generate_dis_isr_parton_dataset.py"
  "scripts/analysis/jet_hadron_observables_altered_parquet_filtered.py"
  "scripts/analysis/jet_hadron_observables_split_pi_pm.py"
  "scripts/analysis/jet_hadron_pi_pm_figures.py"
  "scripts/analysis/merge_editable_source_v1_chunks.py"
  "scripts/analysis/modify_dis_isr_parton_dataset.py"
  "scripts/analysis/plot_eta_xq_valence_lowQ_overlay_two_beams.py"
  "scripts/analysis/plot_jet_hadron_pi_pm_observables.py"
  "scripts/analysis/split_dis_sample_diquark_kick_v2.py"
  "scripts/analysis/transform_editable_all_altered_parquet.py"
  "scripts/generation/generate_events_raw.py"
  "src/diquark/__init__.py"
  "src/diquark/analyze_events_raw.py"
  "src/diquark/paths.py"
)

# --- Parse args ------------------------------------------------------------
APPLY=0
for arg in "$@"; do
  case "$arg" in
    --apply) APPLY=1 ;;
    -h|--help)
      grep '^#' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) echo "Unknown argument: $arg" >&2; exit 2 ;;
  esac
done

# --- Keep-test (bash 3.2 compatible: no associative arrays) ----------------
# Returns 0 if the given absolute path is a manifest file or this script.
is_kept() {
  local target="$1" rel
  [[ "$target" == "${ROOT}/${SELF}" ]] && return 0
  for rel in "${MANIFEST[@]}"; do
    [[ "$target" == "${ROOT}/${rel}" ]] && return 0
  done
  return 1
}

# --- Collect deletion candidates ------------------------------------------
# Every file under ROOT except those inside .git/ and except manifest/self.
to_delete=()
while IFS= read -r -d '' f; do
  if ! is_kept "$f"; then
    to_delete+=("$f")
  fi
done < <(find "$ROOT" -type f -not -path "${ROOT}/.git/*" -print0)

# --- Report / act ----------------------------------------------------------
if [[ ${#to_delete[@]} -eq 0 ]]; then
  echo "Tree is already clean — nothing to remove."
  exit 0
fi

echo "Repo root: $ROOT"
echo "Files not in manifest: ${#to_delete[@]}"
echo
for f in "${to_delete[@]}"; do
  printf '  %s\n' "${f#"$ROOT"/}"
done
echo

if [[ $APPLY -eq 0 ]]; then
  echo "DRY RUN — nothing deleted. Re-run with --apply to remove the files above."
  exit 0
fi

echo "Deleting ${#to_delete[@]} file(s)..."
for f in "${to_delete[@]}"; do
  rm -f -- "$f"
done

# Remove now-empty directories (leaves .git/ and the source dirs intact).
find "$ROOT" -type d -empty -not -path "${ROOT}/.git/*" -not -path "$ROOT" -delete 2>/dev/null || true

echo "Done. Tree reduced to the manifest set."

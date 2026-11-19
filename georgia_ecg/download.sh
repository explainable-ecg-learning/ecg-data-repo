#!/usr/bin/env bash
# Download Georgia 12-Lead ECG database (CinC 2020 Challenge, training/georgia).
# Uses per-subfolder RECORDS files to build an exact download list — avoids
# the 404 noise that wget recursive mode produces on PhysioNet HTML listings.

set -euo pipefail

DEST_DIR="$(cd "$(dirname "$0")" && pwd)/data"
BASE_URL="https://physionet.org/files/challenge-2020/1.0.2/training/georgia"

if [ -f "${DEST_DIR}/RECORDS" ] && [ "$(wc -l < "${DEST_DIR}/RECORDS")" -ge 10344 ]; then
    echo "Georgia ECG already present at ${DEST_DIR} ($(wc -l < "${DEST_DIR}/RECORDS") records), skipping download."
    exit 0
fi

echo "Downloading Georgia 12-Lead ECG database to ${DEST_DIR} ..."
mkdir -p "${DEST_DIR}"

# Fetch the list of subfolders from the top-level challenge RECORDS manifest.
SUBFOLDERS=$(curl -sf "https://physionet.org/files/challenge-2020/1.0.2/RECORDS" \
    | grep "^training/georgia/" \
    | sed 's|^training/georgia/||; s|/$||')

# Rebuild RECORDS file (may already exist if this is a resumed download).
: > "${DEST_DIR}/RECORDS"

for sub in ${SUBFOLDERS}; do
    echo "  Downloading subfolder ${sub} ..."
    mkdir -p "${DEST_DIR}/${sub}"

    # Fetch the per-subfolder RECORDS file to get exact record names.
    SUB_RECORDS=$(curl -sf "${BASE_URL}/${sub}/RECORDS")

    for rec in ${SUB_RECORDS}; do
        echo "${sub}/${rec}" >> "${DEST_DIR}/RECORDS"
        for ext in hea mat; do
            dest="${DEST_DIR}/${sub}/${rec}.${ext}"
            if [ ! -f "${dest}" ]; then
                wget -q -c "${BASE_URL}/${sub}/${rec}.${ext}" -O "${dest}" \
                    || { echo "  WARNING: failed to download ${sub}/${rec}.${ext}"; rm -f "${dest}"; }
            fi
        done
    done
done

echo "Georgia ECG download complete ($(wc -l < "${DEST_DIR}/RECORDS") records)."

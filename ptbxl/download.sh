#!/usr/bin/env bash
# Download PTB-XL dataset (PhysioNet v1.0.3) and soft segmentations (Zenodo 7610236).

set -euo pipefail

DEST_DIR="$(dirname "$0")/data"
PHYSIONET_URL="https://physionet.org/files/ptb-xl/1.0.3"
ZENODO_SEG_BASE="https://zenodo.org/api/records/7610236/files"

mkdir -p "${DEST_DIR}"

if [ -f "${DEST_DIR}/ptbxl_database.csv" ]; then
    echo "PTB-XL v1.0.3 already present at ${DEST_DIR}, skipping waveform download."
else
    echo "Downloading PTB-XL 1.0.3 from PhysioNet to ${DEST_DIR} ..."
    wget -r -N -c -np \
        --directory-prefix="${DEST_DIR}" \
        --cut-dirs=3 \
        --no-host-directories \
        "${PHYSIONET_URL}/"
fi

if [ -f "${DEST_DIR}/ptbxl_segmentations_8bit.zip" ]; then
    echo "PTB-XL soft segmentations already present at ${DEST_DIR}/ptbxl_segmentations_8bit.zip, skipping."
else
    echo "Downloading PTB-XL soft segmentations (Zenodo 7610236) to ${DEST_DIR} ..."
    wget -c "${ZENODO_SEG_BASE}/README/content" -O "${DEST_DIR}/README_segmentations.txt"
    wget -c "${ZENODO_SEG_BASE}/ptbxl_segmentations_8bit.zip/content" -O "${DEST_DIR}/ptbxl_segmentations_8bit.zip"
fi

echo "PTB-XL download complete."

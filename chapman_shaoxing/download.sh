#!/usr/bin/env bash
# Download Chapman-Shaoxing ECG database (PhysioNet ecg-arrhythmia 1.0.0).

set -euo pipefail

DEST_DIR="$(dirname "$0")/data"
PHYSIONET_URL="https://physionet.org/files/ecg-arrhythmia/1.0.0"

if [ -f "${DEST_DIR}/RECORDS" ]; then
    echo "Chapman-Shaoxing already present at ${DEST_DIR}, skipping download."
    exit 0
fi

echo "Downloading Chapman-Shaoxing ECG database to ${DEST_DIR} ..."
echo "NOTE: Total uncompressed size is ~5 GB."
mkdir -p "${DEST_DIR}"

wget -r -N -c -np \
    --directory-prefix="${DEST_DIR}" \
    --cut-dirs=3 \
    --no-host-directories \
    "${PHYSIONET_URL}/"

echo "Download complete."

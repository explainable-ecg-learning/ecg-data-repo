#!/usr/bin/env bash
# Download MIMIC-IV-ECG v1.0 from PhysioNet if not already present.
# Access is open (no credentialling required).

set -euo pipefail

DEST_DIR="$(dirname "$0")/data"
PHYSIONET_URL="https://physionet.org/files/mimic-iv-ecg/1.0"

if [ -f "${DEST_DIR}/record_list.csv" ]; then
    echo "MIMIC-IV-ECG already present at ${DEST_DIR}, skipping download."
    exit 0
fi

echo "Downloading MIMIC-IV-ECG 1.0 from PhysioNet to ${DEST_DIR} ..."
echo "NOTE: Total uncompressed size is ~90 GB. This may take a long time."
mkdir -p "${DEST_DIR}"

wget -r -N -c -np \
    --directory-prefix="${DEST_DIR}" \
    --cut-dirs=3 \
    --no-host-directories \
    "${PHYSIONET_URL}/"

echo "Download complete."

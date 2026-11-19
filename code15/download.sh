#!/usr/bin/env bash
# Download CODE-15% ECG dataset from Zenodo (18 HDF5 part files + exams.csv).

set -euo pipefail

DEST_DIR="$(dirname "$0")/data"
ZENODO_BASE="https://zenodo.org/records/4916206/files"

if [ -f "${DEST_DIR}/exams.csv" ]; then
    echo "CODE-15% already present at ${DEST_DIR}, skipping download."
    exit 0
fi

echo "Downloading CODE-15% ECG dataset to ${DEST_DIR} ..."
echo "NOTE: Total size is ~46 GB (18 zip files). This may take a long time."
mkdir -p "${DEST_DIR}"

# Download metadata CSV
wget -N -c -P "${DEST_DIR}" "${ZENODO_BASE}/exams.csv?download=1" -O "${DEST_DIR}/exams.csv"

# Download and extract each part
for i in $(seq 0 17); do
    ZIP_FILE="${DEST_DIR}/exams_part${i}.zip"
    HDF5_FILE="${DEST_DIR}/exams_part${i}.hdf5"

    if [ -f "${HDF5_FILE}" ]; then
        echo "Part ${i} already extracted, skipping."
        continue
    fi

    echo "Downloading part ${i} of 17..."
    wget -N -c -P "${DEST_DIR}" "${ZENODO_BASE}/exams_part${i}.zip?download=1" -O "${ZIP_FILE}"

    echo "Extracting part ${i}..."
    unzip -o "${ZIP_FILE}" -d "${DEST_DIR}"
    rm "${ZIP_FILE}"
done

echo "Download and extraction complete."

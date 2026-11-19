#!/usr/bin/env bash
# Download all ECG datasets.

set -euo pipefail

SCRIPT_DIR="$(dirname "$0")"

echo "=== Downloading all ECG datasets ==="
echo ""

bash "${SCRIPT_DIR}/ptbxl/download.sh"
echo ""

bash "${SCRIPT_DIR}/mimic_iv_ecg/download.sh"
echo ""

bash "${SCRIPT_DIR}/chapman_shaoxing/download.sh"
echo ""

bash "${SCRIPT_DIR}/georgia_ecg/download.sh"
echo ""

bash "${SCRIPT_DIR}/code15/download.sh"
echo ""

echo "=== All downloads complete ==="

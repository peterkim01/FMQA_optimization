#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
#   bash scripts/setup_local_editable.sh
python -m pip install --no-build-isolation -e ./fmqa
python -m pip install --no-build-isolation -e ./ECP
python -m pip install --no-build-isolation -e ./bbo_via_fmqa

echo "Install complete."


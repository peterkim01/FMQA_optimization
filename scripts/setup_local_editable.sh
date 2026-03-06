#!/usr/bin/env bash
set -euo pipefail

# Run from repository root:
#   bash scripts/setup_local_editable.sh

for project in fmqa ECP bbo_via_fmqa; do
    if [[ -d "./${project}" ]]; then
        python -m pip install --no-build-isolation -e "./${project}"
    else
        echo "Skipping ${project}: directory not present."
    fi
done

echo "Install complete."

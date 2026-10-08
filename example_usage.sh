#!/bin/bash
#
# somospie-pegasus-workflow - example usage walkthrough.
#
# Lightweight demos that run as-is. The full pipeline lives in
# run_manual.sh; Pegasus submission commands are shown commented.
#
# Prerequisites:
#   conda env create -f Apptainer/environment.yml && conda activate somospie
#   pip install -r requirements.txt          # Pegasus.api + PyYAML
#   # Linux only: apptainer build Apptainer/SOMOSPIE_Container.sif \
#   #     Apptainer/SOMOSPIE_Container.def   (cannot build on macOS)

# ==============================================================================
# 1. Ask the terrain service for presigned URLs (no credentials needed)
# ==============================================================================

curl -s "https://intersect.nationalsciencedatafabric.org/somospie/api/v1/somospie/gen-url?state=TN&resolution=30m" \
  | head -c 400
echo

# ==============================================================================
# 2. Fetch one state cutline (1.3 MB source; exercises acquire + cache)
# ==============================================================================

./bin/acquire_roi.py \
    --gadm-url https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_USA_1.json.zip \
    --state TN \
    --output-dir output/example --output-zip output/example/state_TN.zip \
    --manifest output/example/roi_manifest.json \
    --cache-dir output/example/cache

# ==============================================================================
# 3. Generate (not submit) Pegasus workflows
# ==============================================================================

echo "=== Generating default (TN) workflow ==="
./workflow_generator.py --output workflow.yml

echo "=== Alabama variant (config untouched, --state override) ==="
./workflow_generator.py --state AL --output workflow_AL.yml

echo "=== Containerized variant (Linux: build the .sif first) ==="
# ./workflow_generator.py --container-sif Apptainer/SOMOSPIE_Container.sif \
#     --output workflow_container.yml

# ==============================================================================
# 4. Submit and monitor (uncomment when ready; needs HTCondor running)
# ==============================================================================

# pegasus-plan --submit -s condorpool -o local workflow.yml
# pegasus-status <run_directory>     # path printed by pegasus-plan
# pegasus-statistics <run_directory>
# pegasus-analyzer <run_directory>   # only if something fails
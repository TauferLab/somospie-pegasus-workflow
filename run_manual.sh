#!/bin/bash
#
# somospie-pegasus-workflow - local end-to-end test (no Pegasus/HTCondor).
# Mirrors config.yml defaults (TN, 2010-01, 2 tiles) and the exact argv
# workflow_generator.py uses per job. Run from the repo root with the
# somospie conda env activated.
set -e

OUT="output/manual"
mkdir -p "${OUT}"

STATE="TN"
RESOLUTION="30m"
SERVICE_URL="https://intersect.nationalsciencedatafabric.org/somospie"
GADM_URL="https://geodata.ucdavis.edu/gadm/gadm4.1/json/gadm41_USA_1.json.zip"
BASE_URL="https://data.ceda.ac.uk/neodc/esacci/soil_moisture/data/daily_files/COMBINED"
YEAR="2010"
MONTH="1"

echo "=== Step 1a: Acquiring terrain predictors ==="
./bin/acquire_terrain.py \
    --service-url "${SERVICE_URL}" \
    --state "${STATE}" --resolution "${RESOLUTION}" \
    --want aspect=aspect.tif \
    --want elevation=elevation.tif \
    --want hillshading=hillshade.tif \
    --want slope=slope.tif \
    --output-dir "${OUT}/terrain" \
    --manifest "${OUT}/terrain_manifest.json"


echo "=== Step 1b: Acquiring ROI cutline (GADM state boundary) ==="
./bin/acquire_roi.py \
    --gadm-url "${GADM_URL}" \
    --state "${STATE}" \
    --output-dir "${OUT}/terrain" \
    --output-zip "${OUT}/terrain/state_${STATE}.zip" \
    --manifest "${OUT}/roi_manifest.json" \
    --cache-dir "${OUT}/cache"


echo "=== Step 1c: Acquiring soil moisture (ESA CCI, ${YEAR}-${MONTH}) ==="
./bin/acquire_soil_moisture.py \
    --base-url "${BASE_URL}" \
    --version 7.1 --year "${YEAR}" --month "${MONTH}" \
    --output-dir "${OUT}/sm" \
    --days-zip "${OUT}/sm_${YEAR}_0${MONTH}_days.zip" \
    --manifest "${OUT}/sm_${YEAR}_0${MONTH}_manifest.json"


echo "=== Step 2a: Monthly soil-moisture raster ==="
./bin/process_soil_moisture.py \
    --days-zip "${OUT}/sm_${YEAR}_0${MONTH}_days.zip" \
    --month-manifest "${OUT}/sm_${YEAR}_0${MONTH}_manifest.json" \
    --output "${OUT}/sm_${YEAR}_0${MONTH}.tif"


echo "=== Step 2b: Processing terrain + cutline ==="
./bin/process_terrain.py \
    --input-dir "${OUT}/terrain" \
    --param-names aspect elevation hillshading slope \
    --param-files aspect.tif elevation.tif hillshade.tif slope.tif \
    --roi-zip "state_${STATE}.zip" \
    --output-dir "${OUT}/proc" \
    --manifest "${OUT}/proc_manifest.json"


echo "=== Step 2c: Training raster ==="
./bin/generate_train.py \
    -i "${OUT}/sm_${YEAR}_0${MONTH}.tif" \
    -f "${OUT}/proc/aspect.tif" "${OUT}/proc/elevation.tif" \
       "${OUT}/proc/hillshade.tif" "${OUT}/proc/slope.tif" \
    -p aspect elevation hillshading slope \
    -s "${OUT}/proc/state_${STATE}_cutline.zip" \
    -o "${OUT}/train_${YEAR}_0${MONTH}.tif"


echo "=== Step 2d: Eval tiles ==="
./bin/generate_eval.py \
    -i "${OUT}/proc/aspect.tif" "${OUT}/proc/elevation.tif" \
       "${OUT}/proc/hillshade.tif" "${OUT}/proc/slope.tif" \
    -p aspect elevation hillshading slope \
    -n 6 --tile 0 0 --tile 0 1 \
    -s "${OUT}/proc/state_${STATE}_cutline.zip" \
    --output-dir "${OUT}"


echo "=== Step 3: Training RandomForest ==="
./bin/train_model.py \
    -i "${OUT}/train_${YEAR}_0${MONTH}.tif" \
    -o "${OUT}/model.pkl" -s "${OUT}/scaler.pkl" -m "${OUT}/metrics.json" \
    --max-trees 500 --n-iter 10 --cv 5 --seed 1024
cat "${OUT}/metrics.json"


echo "=== Step 4: Inference (2 tiles) ==="
./bin/evaluate_model.py \
    -i "${OUT}/eval_0000.tif" -o "${OUT}/predictions_0000.tif" \
    -s "${OUT}/scaler.pkl" -m "${OUT}/model.pkl"
./bin/evaluate_model.py \
    -i "${OUT}/eval_0001.tif" -o "${OUT}/predictions_0001.tif" \
    -s "${OUT}/scaler.pkl" -m "${OUT}/model.pkl"


echo "=== Step 5: State map ==="
./bin/merge_predictions.py \
    -i "${OUT}/predictions_0000.tif" "${OUT}/predictions_0001.tif" \
    -o "${OUT}/predictions_${STATE}_${YEAR}_0${MONTH}_map.tif"


echo "=== Step 6: Figure ==="
./bin/visualize_map.py \
    --input "${OUT}/predictions_${STATE}_${YEAR}_0${MONTH}_map.tif" \
    --output "${OUT}/map_${STATE}_${YEAR}_0${MONTH}.png" \
    --cmap RdBu --dpi 150 \
    --title "Predicted soil moisture — ${STATE} ${YEAR}-0${MONTH}"
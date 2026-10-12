#!/bin/bash

# Default values
OUTPUT_BASE=${1:-"smoothed_1d_verycomp_sel/{bkg.era.name}"}
PYTHON=".venv/bin/python"

# Signal versions and categories
YEARS=${2:-"2018"}  # Comma-separated or space-separated years
VERSIONS=("Signal312VeryComp" "Signal313VeryComp")
CATEGORIES=("comp")

# Convert YEARS to an array
IFS=', ' read -r -a YEAR_ARRAY <<< "$YEARS"

# Ensure we're in the right directory
ROOT_DIR=$(pwd)

for YEAR in "${YEAR_ARRAY[@]}"; do
    for VERSION in "${VERSIONS[@]}"; do
        for CATEGORY in "${CATEGORIES[@]}"; do
            echo "Processing year: $YEAR, version: $VERSION, category: $CATEGORY"
            
            BG_FILE="2026_09_10__dbs_verycomp_sel_w_1d_dump/${VERSION}/${YEAR}/sm_bkg/${CATEGORY}_mStop.pklz4"

            CONFIG="resources/smoothing_configs_1d/config_${CATEGORY}_${VERSION}.yaml"
            OUT_DIR="${OUTPUT_BASE}/${VERSION}/${CATEGORY}"
            
            if [ ! -f "$BG_FILE" ]; then
                echo "Skipping: $BG_FILE not found."
                continue
            fi

            echo "Running fitting..."

            OUT_PATH=$($PYTHON -m fitting resolve-output \
                --config "$CONFIG" \
                --background "$BG_FILE" \
                --output-format "$OUT_DIR")

            if [ -z "$OUT_PATH" ]; then
                echo "Error:  output path not captured."
                continue
            fi

            echo "OUTPATH IS $OUT_PATH"

            echo $PYTHON -m fitting run --config "$CONFIG" --background "$BG_FILE" --output "$OUT_DIR"
            $PYTHON -m fitting run \
                --config "$CONFIG" \
                --background "$BG_FILE" \
                --output "$OUT_DIR"


            # 2. Run the smoothing
            SMOOTHED_OUTPUT="${OUT_PATH}"
            echo "Running smoothing..."
            $PYTHON -m fitting smooth \
                --state "$OUT_PATH" \
                --name "$CATEGORY" \
                --output-dir "$SMOOTHED_OUTPUT" \
                --num-samples 100 \
                --scale-to "2026_09_10__dbs_verycomp_sel_w_1d_dump/${VERSION}/${YEAR}/data/${CATEGORY}_mStop.pklz4"

            echo "Completed: $CATEGORY for $YEAR/$VERSION"
            echo "----------------------------------------"
        done
    done
done

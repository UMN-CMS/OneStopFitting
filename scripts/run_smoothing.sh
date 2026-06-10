#!/bin/bash

# Default values
OUTPUT_BASE=${1:-"smoothed_complete_backgrounds/{era.name}/{pipeline}/"}
YEARS=${2:-"2016,2017,2018,run3"}  # Comma-separated or space-separated years
PYTHON=".venv/bin/python"

# Signal versions and categories
VERSIONS=("Signal313" "Signal312")
CATEGORIES=("comp" "uncomp" "verycomp")

# Convert YEARS to an array
IFS=', ' read -r -a YEAR_ARRAY <<< "$YEARS"

# Ensure we're in the right directory
ROOT_DIR=$(pwd)

for YEAR in "${YEAR_ARRAY[@]}"; do
    for VERSION in "${VERSIONS[@]}"; do
        for CATEGORY in "${CATEGORIES[@]}"; do
            echo "Processing year: $YEAR, version: $VERSION, category: $CATEGORY"
            
            BG_FILE="combined_backgrounds/$VERSION/${YEAR}_${CATEGORY}_mStop_vs_mChiRatio.pklz4"
            CONFIG="resources/smoothing_configs/$VERSION/${CATEGORY}.yaml"
            OUT_DIR="${OUTPUT_BASE}/${CATEGORY}"
            
            if [ ! -f "$BG_FILE" ]; then
                echo "Skipping: $BG_FILE not found."
                continue
            fi

            echo "Running fitting..."
            echo $PYTHON -m fitting run --config "$CONFIG" --background "$BG_FILE" --output "$OUT_DIR"

            OUT_PATH=$($PYTHON -m fitting resolve-output \
                --config "$CONFIG" \
                --background "$BG_FILE" \
                --output-format "$OUT_DIR")

            echo "OUTPATH IS $OUT_PATH"

            $PYTHON -m fitting run \
                --config "$CONFIG" \
                --background "$BG_FILE" \
                --output "$OUT_DIR"

            if [ -z "$OUT_PATH" ]; then
                echo "Error: Fitting failed or output path not captured."
                continue
            fi

            # 2. Run the smoothing
            SMOOTHED_OUTPUT="${OUT_PATH}"
            echo "Running smoothing..."
            $PYTHON -m fitting smooth \
                --state "$OUT_PATH" \
                --name "$CATEGORY" \
                --output-dir "$SMOOTHED_OUTPUT" \
                --num-samples 100

            echo "Completed: $CATEGORY for $YEAR/$VERSION"
            echo "----------------------------------------"
        done
    done
done

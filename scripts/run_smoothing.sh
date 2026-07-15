#!/bin/bash

# Default values
OUTPUT_BASE=${1:-"smoothed_complete_backgrounds/{era.name}/{pipeline}/"}
PYTHON=".venv/bin/python"

# Signal versions and categories
YEARS=${2:-"2016,2017,2018,Run3"}  # Comma-separated or space-separated years
# VERSIONS=("Signal313" "Signal312")
# CATEGORIES=("comp" "uncomp" "verycomp")

# YEARS=${2:-"2016"}  # Comma-separated or space-separated years
VERSIONS=("Signal313" )
CATEGORIES=("comp" "uncomp" "verycomp")

# Convert YEARS to an array
IFS=', ' read -r -a YEAR_ARRAY <<< "$YEARS"

# Ensure we're in the right directory
ROOT_DIR=$(pwd)

for YEAR in "${YEAR_ARRAY[@]}"; do
    for VERSION in "${VERSIONS[@]}"; do
        for CATEGORY in "${CATEGORIES[@]}"; do
            echo "Processing year: $YEAR, version: $VERSION, category: $CATEGORY"
            
            BG_FILE="combined_backgrounds/$VERSION/${YEAR}/sm_bkg/${CATEGORY}_mStop_vs_mChiRatio.pklz4"
            CONFIG="resources/smoothing_configs/config_${CATEGORY}_${VERSION}.yaml"
            OUT_DIR="${OUTPUT_BASE}/${CATEGORY}"
            
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
                --scale-to "combined_backgrounds/${VERSION}/${YEAR}/data/${CATEGORY}_mStop_vs_mChiRatio.pklz4"

            echo "Completed: $CATEGORY for $YEAR/$VERSION"
            echo "----------------------------------------"
        done
    done
done

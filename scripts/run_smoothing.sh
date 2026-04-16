#!/bin/bash

# Default values
DATASET=${1:-qcd_inclusive_2018}
OUTPUT_BASE=${2:-"smoothed2/{era.name}/{pipeline}/{dataset_name}"}
YEARS=${3:-"2018"}  # Comma-separated or space-separated years
PYTHON=".venv/bin/python"

# Signal versions and categories
VERSIONS=("Signal313" "Signal312")
CATEGORIES=("comp" "uncomp" "verycomp")

# Convert YEARS to an array
IFS=', ' read -r -a YEAR_ARRAY <<< "$YEARS"

# Ensure we're in the right directory
ROOT_DIR=$(pwd)

for YEAR in "${YEAR_ARRAY[@]}"; do
    # Replace '2018' in the dataset name with the current year if it's there
    # This assumes the user passed a dataset name like 'qcd_inclusive_2018'
    CURRENT_DATASET="${DATASET//2018/$YEAR}"
    
    for VERSION in "${VERSIONS[@]}"; do
        for CATEGORY in "${CATEGORIES[@]}"; do
            echo "Processing year: $YEAR, version: $VERSION, category: $CATEGORY for dataset: $CURRENT_DATASET"
            
            # Construct paths
            BG_FILE="export2018nosyst/$YEAR/$VERSION/$CURRENT_DATASET/${CATEGORY}_mStop_vs_mChiRatio.pklz4"
            CONFIG="resources/smoothing_configs/$VERSION/${CATEGORY}.yaml"
            OUT_DIR="${OUTPUT_BASE}/${CATEGORY}"
            
            if [ ! -f "$BG_FILE" ]; then
                echo "Skipping: $BG_FILE not found."
                continue
            fi

            echo "Running fitting..."
            echo $PYTHON -m fitting run --config "$CONFIG" --background "$BG_FILE" --output "$OUT_DIR"

            OUT_PATH=$($PYTHON -m fitting run \
                --config "$CONFIG" \
                --background "$BG_FILE" \
                --output "$OUT_DIR" | grep "FITTING_OUTPUT_PATH:" | cut -d' ' -f2)

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
                --num-samples 200

            echo "Completed: $CATEGORY for $YEAR/$VERSION"
            echo "----------------------------------------"
        done
    done
done

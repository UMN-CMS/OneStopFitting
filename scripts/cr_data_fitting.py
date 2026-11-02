OUTPUT_BASE=${1:-"results/2026_10_30__cr_test/{bkg.era.name}"}
PYTHON=".venv/bin/python"

# Signal versions and categories
YEARS=${2:-"2016, 2018, 2017, 2022_preEE, Run3"}  # Comma-separated or space-separated years
VERSIONS=("Control")
CATEGORIES=("uncomp" "comp" "verycomp")

# Convert YEARS to an array
IFS=', ' read -r -a YEAR_ARRAY <<< "$YEARS"

# Ensure we're in the right directory
ROOT_DIR=$(pwd)

for YEAR in "${YEAR_ARRAY[@]}"; do
    for VERSION in "${VERSIONS[@]}"; do
        for CATEGORY in "${CATEGORIES[@]}"; do
            echo "Processing year: $YEAR, version: $VERSION, category: $CATEGORY"

            BG_FILE="export_combined_backgrounds/${VERSION}/${YEAR}/data/${CATEGORY}_mStop_vs_mChiRatio.pklz4"
            CONFIG="resources/new_opt/config_${CATEGORY}_${VERSION}.yaml"
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

            echo "Completed: $CATEGORY for $YEAR/$VERSION"
            echo "----------------------------------------"
        done
    done
done

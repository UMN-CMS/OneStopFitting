OUTPUT_FORMAT="$OUTPUT/{era.name}/{dataset_name}/{injection_rate}"

# Resolve the actual output directory path using metadata from input files and config
ACTUAL_OUTPUT=$(python3 -m fitting resolveoutput \
    --background "$BACKGROUND" \
    --signal "$SIGNAL" \
    --config "$CONFIG" \
    --output-format "$OUTPUT_FORMAT")

echo "Resolved output directory: $ACTUAL_OUTPUT"

# Run the pipeline
python3 -m fitting run --background  $BACKGROUND \
        --signal $SIGNAL \
        --output "$ACTUAL_OUTPUT" \
        --config "$CONFIG"

# Run combine commands if script was generated
COMBINE_SCRIPT="$ACTUAL_OUTPUT/combine/run_combine_commands.sh"
if [ -f "$COMBINE_SCRIPT" ]; then
    echo "Running combine commands from: $COMBINE_SCRIPT"
    bash "$COMBINE_SCRIPT"
else
    echo "No combine script found at: $COMBINE_SCRIPT"
fi

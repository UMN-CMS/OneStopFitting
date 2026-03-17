python3 -m fitting run --background  $BACKGROUND \
        --signal $SIGNAL \
        --output "$OUTPUT/{era.name}/{dataset_name}/{injection_rate}" \
        --config "$CONFIG"
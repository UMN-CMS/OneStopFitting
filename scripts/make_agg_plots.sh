#!/bin/bash


INPUT_JSON=$1
OUTPUT_DIR=$2

# python3 -m fitting aggregate   -o $OUTPUT_DIR \
#     --group-by metadata.other_data.coupling \
#     --group-by metadata.era.name \
#     --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --cmin -2 --cmax 2 --plot-types scatter \
#     -m combine.multidim_fit.r -m combine.multidim_fit.r_err \
#     --xlim -10,10 --vlines 0.0 $INPUT_JSON

# python3 -m fitting aggregate   -o $OUTPUT_DIR \
#     --group-by metadata.other_data.coupling \
#     --group-by metadata.era.name \
#     --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --cmin -2 --cmax 2 --plot-types scatter \
#     -m combine.gof_p_value \
#     --xlim 0,1 --vlines 0.5 $INPUT_JSON

# python3 -m fitting aggregate   -o $OUTPUT_DIR \
#     --group-by metadata.other_data.coupling \
#     --group-by metadata.era.name \
#     --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --cmin -2 --cmax 2 --plot-types scatter \
#     -m ppc.test_stats.chi2.blinded.pvalue \
#     --xlim 0,1 --vlines 0.5 $INPUT_JSON

python3 -m fitting aggregate   -o $OUTPUT_DIR \
    --group-by metadata.other_data.coupling \
    --group-by metadata.era.name \
    --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
    --cmin 0.5 --cmax 1.5  --cmap coolwarm \
    --plot-types scatter --plot-types  mass_plane \
    -m metrics.blinded_chi2_per_bin \
    --xlim 0,2 --vlines 1.0 $INPUT_JSON



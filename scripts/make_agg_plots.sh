#!/bin/bash


INPUT_JSON=$1
OUTPUT_DIR=$2
python3 -m fitting aggregate   -o $OUTPUT_DIR \
    --group-by metadata.other_data.coupling \
    --group-by metadata.era.name \
    --group-by config.injection_rate \
    -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
    --plot-types mass_plane \
    --plot-types mass_plane_smooth \
    --transform r_to_coupling \
    --draw-contours 0.1,0.2,0.3,0.4 \
    --contour-fmt "$\lambda_{{{other_data.coupling}}}=$%.1f " \
    --smooth-sigma 8 \
    -m combine.limits.expected \
    --colorbar-label 'Expected 95% CL Limit $\lambda_{{{other_data.coupling}}}^{{\prime\prime}}$' \
    --cms-extra '$\tilde{{t}} \to b \tilde{{\chi}}^{{\pm}}\to 4j $' \
    --xlim 0,1 --vlines 0.5 $INPUT_JSON

# python3 -m fitting aggregate   -o $OUTPUT_DIR \
#     --group-by metadata.other_data.coupling \
#     --group-by metadata.era.name \
#     --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --cmin -2 --cmax 2 --plot-types scatter \
#     --plot-types report \
#     -m combine.gof_p_value \
#     --xlim 0,1 --vlines 0.5 $INPUT_JSON

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
#     -m combine.multidim_fit.r -m combine.multidim_fit.r_err \
#     --xlim -10,10 --vlines 0.0 $INPUT_JSON


# python3 -m fitting aggregate   -o $OUTPUT_DIR \
#     --group-by metadata.other_data.coupling \
#     --group-by metadata.era.name \
#     --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --cmin -2 --cmax 2 --plot-types scatter \
#     -m ppc.test_stats.chi2.blinded.pvalue \
#     --xlim 0,1 --vlines 0.5 $INPUT_JSON

# python3 -m fitting aggregate   -o $OUTPUT_DIR \
#     --group-by metadata.other_data.coupling \
#     --group-by metadata.era.name \
#     --group-by config.injection_rate   -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --cmin 0.5 --cmax 1.5  --cmap coolwarm \
#     --plot-types scatter --plot-types  mass_plane \
#     -m metrics.blinded_chi2_per_bin \
#     --xlim 0,2 --vlines 1.0 $INPUT_JSON



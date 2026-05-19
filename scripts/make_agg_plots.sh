#!/bin/bash


INPUT_JSON=$1
OUTPUT_DIR=$2

SHARED_OPTS=(
    -i "$INPUT_JSON"
    -o "$OUTPUT_DIR"
    --group-by metadata.other_data.coupling
    --group-by metadata.era.name
    --group-by config.injection_rate
)

# Mass plane + smooth: expected significance
python3 -m fitting aggregate "${SHARED_OPTS[@]}" \
    mass-plane -m combine.significance \
    -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
    --draw-contours 3,5 \
    --contour-fmt '$\sigma=%.1f$' \
    --colorbar-label 'Expected Significance' \
    --cms-extra '$\tilde{{t}} \to b \tilde{{\chi}}^{{\pm}}\to 4j $'

python3 -m fitting aggregate "${SHARED_OPTS[@]}" \
    smooth -m combine.significance \
    -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
    --smooth-sigma 16 \
    --draw-contours 3,5 \
    --contour-fmt '$\sigma=%.1f$' \
    --colorbar-label 'Expected Significance' \
    --cms-extra '$\tilde{{t}} \to b \tilde{{\chi}}^{{\pm}}\to 4j $'

# Mass plane: expected limits with coupling transform
# python3 -m fitting aggregate "${SHARED_OPTS[@]}" \
#     mass-plane -m combine.limits.expected \
#     -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --transform r_to_coupling \
#     --draw-contours 0.1,0.2,0.3,0.4 \
#     --contour-fmt "$\lambda_{{{other_data.coupling}}}=$%.1f " \
#     --colorbar-label 'Expected 95% CL Limit $\lambda_{{{other_data.coupling}}}^{{\prime\prime}}$' \
#     --cms-extra '$\tilde{{t}} \to b \tilde{{\chi}}^{{\pm}}\to 4j $'

# Scatter: gof p-value
# python3 -m fitting aggregate "${SHARED_OPTS[@]}" \
#     scatter -m combine.gof_p_value \
#     -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --xlim 0,1 --vlines 0.5

# Scatter: multidim fit r with errors
# python3 -m fitting aggregate "${SHARED_OPTS[@]}" \
#     scatter -m combine.multidim_fit.r -m combine.multidim_fit.r_err \
#     -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --xlim -10,10 --vlines 0.0

# Scatter: blinded chi2 per bin
# python3 -m fitting aggregate "${SHARED_OPTS[@]}" \
#     scatter -m metrics.blinded_chi2_per_bin \
#     -n '{config.injection_rate}/{metadata.era.name}_{metadata.other_data.coupling}_{metric_name}' \
#     --xlim 0,2 --vlines 1.0

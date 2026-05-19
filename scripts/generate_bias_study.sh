#!/bin/bash



python3 -m fitting makebatch \
    --signal 'export_combined/{year}/{pipeline}/signal*{year}*/{category}*.pklz4' \
    --background 'smoothed_combined/{year}/{pipeline}/qcd_inclusive_2018/{category}/{category}_{toy_index}.pklz4'\
    --years 2018 \
    --pipelines Signal312 --pipelines Signal313 \
    --output results/2026_05_14_bias_study \
    --config-base "resources/no_cut/config_{category}_{pipeline}.yaml"  \
    --subdir-format "{era.name}/{injection_dataset_name}/{dataset_name}/{injection_rate}/{toy_index}"  \
    --num-toys 10 --injection-rates 1.0,0.25  \
    --param "injection_signal_path='export_combined/{year}/{pipeline}/signal_{year}_{coupling}_1000_400_combined/{category}_mStop_vs_mChiRatio.pklz4','export_combined/{year}/{pipeline}/signal_{year}_{coupling}_1700_600_combined/{category}_mStop_vs_mChiRatio.pklz4','export_combined/{year}/{pipeline}/signal_{year}_{coupling}_1400_1300_combined/{category}_mStop_vs_mChiRatio.pklz4','export_combined/{year}/{pipeline}/signal_{year}_{coupling}_1800_1600_combined/{category}_mStop_vs_mChiRatio.pklz4'"

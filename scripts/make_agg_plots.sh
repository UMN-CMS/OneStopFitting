#!/bin/bash

for pipeline in "312" "313"; do 
    for year in 2018; do
        for x in 0p5 0p75 1p0 1p25 1p5 1p75 2p0 2p25; do 
            for metric in ppc.test_stats.chi2.blinded.pvalue; do
                python3 -m fitting aggregate-plot \
                    -m $metric \
                    -o aggplots/$year/$pipeline \
                    -n "{metric}_$x" \
                    "results/2026_03_30_fixed_pipes/$year/*$pipeline*/$x/**/summary.json"
            done
            for metric in metrics.blinded_chi2_per_bin; do
                python3 -m fitting aggregate-plot \
                    -m $metric \
                    -o aggplots/$year/$pipeline \
                    --cmin 0.0 \
                    --cmax 3.0 \
                    -n "{metric}_$x" \
                    results/2026_03_30_fixed_pipes/$year/*$pipeline*/$x/**/summary.json;
            done
        done
    done
done

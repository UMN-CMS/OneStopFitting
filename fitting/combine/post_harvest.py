from __future__ import annotations

import json
import logging
import re
from fitting.diagnostics.plot_utils import plotPPD
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Callable
import numpy as np

logger = logging.getLogger(__name__)


def postHarvest(data):
    if "combine" not in data:
        raise ValueError(
            "Combine data not found in summary file. Make sure to run 'harvest' first"
        )

    ret_data = {}
    ret_plots = {}
    combine_data = data["combine"]

    gof_test_statistic = combine_data.get("gof_test_statistic")
    gof_test_statistic_toys = combine_data.get("gof_test_statistic_toys")

    if gof_test_statistic is not None and gof_test_statistic_toys is not None:
        gof_test_statistic_toys = np.array(gof_test_statistic_toys)
        p_value = np.mean(gof_test_statistic_toys >= gof_test_statistic)
        ret_data["gof_p_value"] = p_value

        fig, ax = plt.subplots()
        plotPPD(
            ax,
            gof_test_statistic_toys,
            gof_test_statistic,
            label="GOF Test Statistic Toys",
        )
        ret_plots["gof_test"] = (fig, ax)

    data["post_harvest"] = ret_data

    return ret_data, ret_plots

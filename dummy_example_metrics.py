"""
#
# Software Name : sstracking_metrics
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE" file for more details or https://opensource.org/license/MIT
#
"""

import logging
from pathlib import Path

import numpy as np

from sstracking_metrics.metric_evaluator import MetricEvaluator
from sstracking_metrics.utils_metrics import Detection, angular_distance

# Configure logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # Configuration parameters
    # ----------------------

    # Output path
    path_results = Path("dummy_examples")

    # Time resolution parameters
    frame_len = 0.032  # Frame length in seconds

    # Similarity measurement parameters
    similarity_params = {
        "threshold_gating": 360,  # in degrees
        "similarity_function": angular_distance,
    }

    # Evaluation parameters
    tolerance_localization_error = 7.5  # in degrees
    bootstrap_params = {
        "bootstrap_times": 20,  # Number of bootstrap iterations
        "bootstrap_rate": 0.8,  # Percentage of data to use in each bootstrap
    }

    # Initialize metric evaluator
    metric_evaluator = MetricEvaluator(
        folder_results=path_results,
        tolerance_localization_error=tolerance_localization_error,
        frame_len=frame_len,
        similarity_params=similarity_params,
        **bootstrap_params,
    )

    # Data Generation and Processing
    # ----------------------------

    # Set total number of frames for each example
    T = 100  # total number of frames per items

    # Example 1: Single source split into 3 tracks
    # Ground truth: constant DOA at 30°
    # Prediction: Three tracks at 30°, 45°, and -20°
    gt1 = np.full((T, 1, 2), 30)  # T, J, 2 (for az, el)
    pr1 = np.full((T, 3, 2), np.nan)
    pr1[: int(T / 4), 0, :] = 30
    pr1[int(T / 4) : int(T / 2), 1, :] = 45
    pr1[int(T / 2) :, 2, :] = -20

    # Example 2: Single source split into 2 tracks
    # Ground truth: constant DOA at 30°
    # Prediction: Two tracks at 20° and 30°
    gt2 = np.full((T, 1, 2), 30)  # T, J, 2 (for az, el)
    pr2 = np.full((T, 2, 2), np.nan)
    pr2[: int(T / 2), 0, :] = 20
    pr2[int(T / 2) :, 1, :] = 30

    # Example 3: Single source split into 4 tracks
    # Ground truth: constant DOA at 30°
    # Prediction: 4 tracks
    gt3 = np.full((T, 1, 2), 30)  # T, J, 2 (for az, el)
    pr3 = np.full((T, 4, 2), np.nan)
    pr3[: int(T / 2), 0, :] = 10
    pr3[: int(T / 2), 1, :] = 30
    pr3[int(T / 2) :, 2, :] = 20
    pr3[int(T / 2) :, 3, :] = 45

    # Organize data
    data_dict = {
        "item_1": {"gt": gt1, "pr": pr1},
        "item_2": {"gt": gt2, "pr": pr2},
        "item_3": {"gt": gt3, "pr": pr3},
    }

    # Prepare Data for Metric Computation
    # ---------------------------------

    ground_truths, predictions = {}, {}
    for i, item in data_dict.items():
        # Extract ground truth and prediction DoAs
        gt_doa = item["gt"]
        pr_doa = item["pr"]

        # Store as Detection objects
        ground_truths[i] = Detection(doa=gt_doa)
        predictions[i] = Detection(doa=pr_doa)

    # Metric Computation
    # -----------------

    # Compute metrics (detection, localization, association)
    metric_evaluator(
        ground_truths,
        predictions,
        name="dummy_example_metrics",
        tocompute=["detection", "localization", "association"],
    )

    # Compute LOCATA metrics separately (slower computation)
    metric_evaluator(
        ground_truths,
        predictions,
        name="dummy_example_metrics",
        tocompute=["locata"],
    )

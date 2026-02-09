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

from sstracking_metrics.metric_evaluator import MetricEvaluator
from sstracking_metrics.utils_dataset import (
    load_json,
    read_array,
    time_to_frame,
)
from sstracking_metrics.utils_metrics import Detection, angular_distance

# Configure logging
logging.basicConfig(level=logging.INFO)


if __name__ == "__main__":
    # Configuration parameters
    # ----------------------

    # Data paths
    root = Path(
        "/home/dxbz2376/Data/librijump/final/"
    )  # change with actual local folder
    path_results = Path("dummy_examples")
    J = 1  # Subset of LibriJump (1, 2 or 3)

    # Audio parameters
    FS = 16000  # Sampling rate in Hz

    # Similarity measurement parameters to compute the matching
    similarity_params = {
        "threshold_gating": 360,  # in degrees
        "similarity_function": angular_distance,
    }

    # Evaluation parameters
    tolerance_localization_error = 7.5  # for the localization accuracy
    bootstrap_params = {
        "bootstrap_times": 20,  # Number of bootstrap iterations
        "bootstrap_rate": 0.8,  # Percentage of data to use in each bootstrap
    }

    # STFT parameters
    spectro_params = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 512,
        "center": True,
        "pad_mode": "constant",
    }
    # Reference: https://librosa.org/doc/0.11.0/generated/librosa.stft.html

    # Data loading and processing
    # --------------------------

    # Set path for specific number of speakers
    path_new = root / f"{J}spk"

    # Load dataset information
    data_dict = load_json(
        path_new / "data.json",
        path_root=str(root),
    )

    # Initialize metric evaluator
    metric_evaluator = MetricEvaluator(
        folder_results=path_results,
        tolerance_localization_error=tolerance_localization_error,
        frame_len=spectro_params["hop_length"] / FS,
        similarity_params=similarity_params,
        **bootstrap_params,
    )

    # Process each item in the dataset
    ground_truths, predictions = {}, {}
    for i, item in data_dict.items():
        # Load ground truth Direction of Arrival (DoA)
        gt_doa = read_array(item["path_doas"])

        # Convert to frame resolution using STFT parameters
        gt_doa = time_to_frame(gt_doa.copy(), spectro_params=spectro_params)

        # Generate predictions (dummy example using ground truth as prediction)
        pr_doa = gt_doa.copy()

        # In a real scenario, replace with actual predictions
        # pr_doa = predictor(mixture)

        # Store ground truths and predictions as Detection objects
        ground_truths[i] = Detection(doa=gt_doa)
        predictions[i] = Detection(doa=pr_doa)

    # Metric Computation
    # -----------------

    # Compute metrics (detection, localization, association)
    metric_evaluator(
        ground_truths,
        predictions,
        name="dummy_example",
        tocompute=["detection", "localization", "association"],
    )

    # Compute LOCATA metrics separately (slower computation)
    metric_evaluator(
        ground_truths,
        predictions,
        name="dummy_example",
        tocompute=["locata"],
    )

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

from typing import Callable

import numpy as np

from .utils_metrics import BaseMetric, Detection


class LocalizationMetric(BaseMetric):
    """
    A metric class for evaluating localization metrics.
    """

    def __init__(
        self,
        threshold_gating: float,
        similarity_function: Callable,
        tolerance: float,
    ):
        """
        Initialize the LocalizationMetric class.

        Args:
            threshold_gating (float): Maximum allowed distance/error for the matchin
            similarity_function (Callable): Function to compute similarity/distance between detections for the matching

            tolerance (float): Angle tolerance in degrees for accuracy calculation
        """
        super().__init__(threshold_gating, similarity_function)
        self.tolerance = tolerance  # degrees

    def run(self, gt_det: Detection, pr_det: Detection):
        """
        Compute localization metrics between ground truth and predictions.

        Args:
            gt_det (Detection): Ground truth detections
            pr_det (Detection): Predicted detections

        Returns:
            dict: Dictionary containing the following metrics:
                - mean: Mean localization error
                - median: Median localization error
                - accuracy_tol: Accuracy within tolerance threshold
                - TP: Number of true positives
        """
        # Extract localization errors for matched detections
        loca_error = self.similarity_matrix[self.matches.astype(bool)]

        # Calculate mean and median localization errors
        mean_loca_error = np.mean(loca_error)
        median_loca_error = np.median(loca_error)

        return {
            "mean": mean_loca_error.copy().item(),
            "median": median_loca_error.copy().item(),
            "accuracy_tol": self.accuracy_tol(loca_error).copy().item(),
            "TP": self.TP.item(),
        }

    def accuracy_tol(self, loca_error: np.ndarray):
        """
        Calculate the localization accuracy (proportion of localization errors that fall within the tolerance threshold).

        Args:
            loca_error (np.ndarray): Array of localization errors

        Returns:
            float: Proportion of errors within tolerance (0 to 1)
        """
        return (loca_error <= self.tolerance).sum() / len(loca_error)

    def get_submetrics(self):
        """
        Get the list of available sub-metrics calculated by this class.
        """
        return ["mean", "median", "accuracy_tol"]

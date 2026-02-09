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


class AssociationMetric(BaseMetric):
    """
    A metric class for evaluating tracking association metrics.

    Based on the TrackEval (https://github.com/JonathonLuiten/TrackEval) implementation of the higher-order tracking accuracy (HOTA) metrics [1].

    [1] Luiten, J., Osep, A., Dendorfer, P., Torr, P., Geiger, A., Leal-Taixé, L., & Leibe, B. (2021).
        Hota: A higher order metric for evaluating multi-object tracking. International journal of computer vision, 129(2), 548-578.

    Args:
        threshold_gating (float): Maximum allowed distance/error for the matchin
        similarity_function (Callable): Function to compute similarity/distance between detections for the matching
    """

    def __init__(self, threshold_gating: float, similarity_function: Callable):
        super().__init__(threshold_gating, similarity_function)

    def run(self, gt_det: Detection, pr_det: Detection):
        """
        Compute association metrics between ground truth and predictions.

        Args:
            gt_det (Detection): Ground truth detections
            pr_det (Detection): Predicted detections

        Returns:
            dict: Dictionary containing the following metrics:
                - accuracy: association accuracy (AssA)
                - precision: association precision (AssPr)
                - recall: association recall (AssRe)
                - TP: Number of true positives
        """
        # Calculate how many times each ground truth ID has been matched
        # Shape: [n_gt_id, 1]
        gt_id_count = self.matches.sum(axis=(0, 2))[:, np.newaxis]

        # Calculate how many times each predicted ID has been matched
        # Shape: [1, n_pr_id]
        pr_id_count = self.matches.sum(axis=(0, 1))[np.newaxis, :]

        # Total number of matches for each ground truth-prediction ID pair
        # Shape: [n_gt_id, n_pr_id]
        matches_count = self.matches.sum(axis=0)

        # Calculate association accuracy (AssA)
        # Formula: matches / (gt_counts + pr_counts - matches)
        ass_a = matches_count / np.maximum(
            1, gt_id_count + pr_id_count - matches_count
        )
        AssA = np.sum(matches_count * ass_a) / np.maximum(1, self.TP)

        # Calculate association recall (AssRe)
        # Formula: matches / gt_counts
        ass_re = matches_count / np.maximum(1, gt_id_count)
        AssRe = np.sum(matches_count * ass_re) / np.maximum(1, self.TP)

        # Calculate association precision (AssPr)
        # Formula: matches / pr_counts
        ass_pr = matches_count / np.maximum(1, pr_id_count)
        AssPr = np.sum(matches_count * ass_pr) / np.maximum(1, self.TP)

        return {
            "accuracy": AssA.copy().item(),
            "precision": AssPr.copy().item(),
            "recall": AssRe.copy().item(),
            "TP": self.TP.item(),
        }

    def get_submetrics(self):
        """
        Get the list of available sub-metrics calculated by this class.
        """
        return ["accuracy", "precision", "recall"]

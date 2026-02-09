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

import numpy as np

from .utils_metrics import BaseMetric, Detection


class DetectionMetric(BaseMetric):
    """
    A metric class for evaluating detection metrics.

    """

    def __init__(self, threshold_gating: float, similarity_function: callable):
        """
        Initialize the DetectionMetric class.

        Args:
            threshold_gating (float): Maximum allowed distance/error for the matchin
            similarity_function (callable): Function to compute similarity/distance between detections for the matching

        """
        super().__init__(threshold_gating, similarity_function)

    def run(self, gt_det: Detection, pr_det: Detection):
        """
        Compute detection metrics between ground truth and predictions.

        Args:
            gt_det (Detection): Ground truth detections
            pr_det (Detection): Predicted detections

        Returns:
            dict: Dictionary containing the following metrics:
                - accuracy: Detection accuracy
                - precision: True positives / (True positives + false positives)
                - recall: True positives / (True positives + false negatives)
                - TP: Number of true positives
                - FP: Number of false positives
                - FN: Number of false negatives
        """
        # Calculate difference between predicted and ground truth detections
        diff = pr_det.nos - gt_det.nos
        # Sum up negative differences for false negatives
        FN = np.abs(diff[diff < 0].sum())
        # Sum up positive differences for false positives
        FP = diff[diff > 0].sum()

        # Calculate metrics
        accuracy = self.TP / (self.TP + FN + FP)
        precision = self.TP / (self.TP + FP)
        recall = self.TP / (self.TP + FN)

        return {
            "accuracy": accuracy.copy().item(),
            "precision": precision.copy().item(),
            "recall": recall.copy().item(),
            "TP": self.TP.item(),
            "FP": FP.item(),
            "FN": FN.item(),
        }

    def combine_scenes(self, metrics_per_scene: dict, metric_name: str):
        """
        Combine metrics from multiple scenes into a single result.

        Args:
            metrics_per_scene (dict): Dictionary containing metrics for each scene
            metric_name (str): Name of the metric to combine

        Returns:
            dict: Dictionary containing combined global metrics:
                - accuracy: Global detection accuracy
                - precision: Global precision
                - recall: Global recall
        """
        # Sum up TPs across all scenes
        global_TP = sum(
            [
                metrics_per_scene[sent][metric_name]["TP"]
                for sent in metrics_per_scene.keys()
            ]
        )
        # Sum up FPs across all scenes
        global_FP = sum(
            [
                metrics_per_scene[sent][metric_name]["FP"]
                for sent in metrics_per_scene.keys()
            ]
        )
        # Sum up FNs across all scenes
        global_FN = sum(
            [
                metrics_per_scene[sent][metric_name]["FN"]
                for sent in metrics_per_scene.keys()
            ]
        )

        # Calculate global metrics using combined values
        global_accuracy = global_TP / (global_TP + global_FN + global_FP)
        global_precision = global_TP / (global_TP + global_FP)
        global_recall = global_TP / (global_TP + global_FN)

        return {
            "accuracy": global_accuracy,
            "precision": global_precision,
            "recall": global_recall,
        }

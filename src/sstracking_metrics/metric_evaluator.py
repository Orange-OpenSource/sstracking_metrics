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
import random
import time
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from .association import AssociationMetric
from .detection import DetectionMetric
from .localization import LocalizationMetric
from .locata import LOCATAMetric
from .utils_metrics import Detection

logger = logging.getLogger(__name__)


class MetricEvaluator:
    """
    A class to evaluate multiple tracking metrics (detection, localization,
    association, and LOCATA metrics) using bootstrap sampling for robust evaluation.
    """

    def __init__(
        self,
        folder_results: Path,
        frame_len: float,
        similarity_params: dict,
        tolerance_localization_error: float,
        bootstrap_times: int,
        bootstrap_rate: float,
    ):
        """
        Initialization with evaluation parameters.

        Args:
            folder_results (Path): Directory to save evaluation results
            frame_len (float): Frame length for LOCATA metric
            similarity_params (dict): Parameters for similarity matrix computation
            tolerance_localization_error (float): tolerance parameter for the localization error
            bootstrap_times (int): Number of bootstrap iterations
            bootstrap_rate (float): Proportion of data to use in each bootstrap sample
        """

        self.metrics = {
            "detection": DetectionMetric(**similarity_params),
            "localization": LocalizationMetric(
                tolerance=tolerance_localization_error, **similarity_params
            ),
            "association": AssociationMetric(**similarity_params),
            "locata": LOCATAMetric(frame_len=frame_len),
        }

        self.folder_results = folder_results
        self.folder_results.mkdir(parents=True, exist_ok=True)

        self.bootstrap_rate = bootstrap_rate
        self.bootstrap_times = bootstrap_times

    def __call__(
        self, ground_truths: dict, predictions: dict, name: str, tocompute: list
    ):
        """
        Main evaluation method that computes metrics using bootstrap sampling.

        Args:
            ground_truths (dict): Dictionary of ground truth detections per scene
            predictions (dict): Dictionary of predicted detections per scene
            name (str): Name identifier for the evaluation results
            tocompute (list): List of metrics to compute
        """
        self.tocompute = tocompute

        self.gt_det = ground_truths
        self.pr_det = predictions

        scenes = list(ground_truths.keys())
        nsent_per_bootstrap = int(self.bootstrap_rate * len(scenes))

        metrics_bootstrap = {}

        logger.info(
            f"Running {self.bootstrap_rate:.0%}-{self.bootstrap_times} bootstrap..."
        )
        logger.info(
            f"Computing the following metrics: {', '.join(self.tocompute)}."
        )

        # Perform bootstrap evaluation
        for bootstrap in tqdm(range(self.bootstrap_times)):
            begin_t = time.time()
            fixed_random = random.Random(bootstrap)  # Set seed

            # bootstrap : sample scenes with replacement
            scenes_bootstrap = fixed_random.choices(
                scenes, k=nsent_per_bootstrap
            )

            metrics_bootstrap[bootstrap] = self.run_one_bootstrap(
                scenes_bootstrap
            )
            logger.info(f"Total time : {time.time() - begin_t:.3f} s")

        # Calculate average results across all bootstraps
        out_values, out_stats = self.avg_metric_results(metrics_bootstrap)

        # Save results to YAML files
        name = f"{name}_{'_'.join(self.tocompute)}"
        with open(
            self.folder_results / f"{name}_results_bootstrap.yaml", "w"
        ) as file:
            yaml.safe_dump(out_values, file, default_flow_style=False)

        with open(
            self.folder_results / f"{name}_stats_bootstrap.yaml", "w"
        ) as file:
            yaml.safe_dump(out_stats, file, default_flow_style=False)

        return

    def run_one_bootstrap(self, scenes: list):
        """
        Run evaluation on one bootstrap sample.

        Args:
            scenes (list): List of scenes to evaluate

        Returns:
            metrics_one_bootstrap: Dictionary containing metrics for this bootstrap iteration
        """
        metrics_per_scene = {}

        for s, scene in enumerate(scenes):
            gt_detection = self.gt_det[scene]
            pr_detection = self.pr_det[scene]

            output = self.run_one_scene(gt_detection, pr_detection)

            # Handle duplicate scenes in bootstrap by adding index
            key = f"{scene}_{s}" if scene in metrics_per_scene.keys() else scene
            metrics_per_scene[key] = output

        # Combine metrics across all scenes
        metrics_one_bootstrap = self.combine_scenes(metrics_per_scene)

        return metrics_one_bootstrap

    def run_one_scene(self, gt_detection: Detection, pr_detection: Detection):
        """
        Compute metrics for a single scene.

        Args:
            gt_detection (Detection): Ground truth detection for the scene
            pr_detection (Detection): Predicted detection for the scene

        Returns:
            Dictionary containing computed metrics for the scene
        """
        output = self.init_dict()

        # Reuse similarity and matches computations across metrics when possible
        similarity, matches = None, None
        for metric in self.tocompute:
            if metric == "locata":
                output[metric] = self.metrics[metric](
                    gt_detection, pr_detection
                )
            else:
                output[metric] = self.metrics[metric](
                    gt_detection, pr_detection, similarity, matches
                )
                if similarity is None or matches is None:
                    similarity = self.metrics[metric].similarity_matrix
                    matches = self.metrics[metric].matches

        return output

    def combine_scenes(self, metrics_per_scene: dict):
        """
        Combine metrics from multiple scenes into a single result.

        Args:
            metrics_per_scene (dict): Dictionary containing metrics for each scene

        Returns:
            dict: Combined metrics across all scenes
        """
        output = self.init_dict()

        for metric in self.tocompute:
            output[metric] = self.metrics[metric].combine_scenes(
                metrics_per_scene, metric
            )

        return output

    def avg_metric_results(self, metrics_bootstrap: dict):
        """
        Calculate average metrics and statistics across all bootstrap iterations.

        Args:
            metrics_bootstrap (dict): Dictionary containing metrics for each bootstrap iteration

        Returns:
            tuple: (out_values, out_stats) where out_values contains all metric values and
                  out_stats contains statistical summaries (mean, std, median)
        """
        out_values = self.init_dict()

        # Collect all values for each metric and submetric
        for bootstrap in metrics_bootstrap.keys():
            for metric in self.tocompute:
                if metric == "locata":
                    # LOCATA metrics are stored in mean/std form
                    for submetric in metrics_bootstrap[bootstrap][
                        metric
                    ].keys():
                        if submetric not in out_values[metric].keys():
                            out_values[metric][submetric] = {
                                "mean": [],
                                "std": [],
                            }

                        out_values[metric][submetric]["mean"].append(
                            metrics_bootstrap[bootstrap][metric][submetric][
                                "mean"
                            ]
                        )
                        out_values[metric][submetric]["std"].append(
                            metrics_bootstrap[bootstrap][metric][submetric][
                                "std"
                            ]
                        )
                else:
                    for submetric in metrics_bootstrap[bootstrap][
                        metric
                    ].keys():
                        if submetric not in out_values[metric].keys():
                            out_values[metric][submetric] = []

                        out_values[metric][submetric].append(
                            metrics_bootstrap[bootstrap][metric][submetric]
                        )

        # Calculate statistics for each metric
        out_stats = self.init_dict()
        for metric in self.tocompute:
            if metric == "locata":
                for submetric in out_values[metric].keys():
                    out_stats[metric][submetric] = {
                        "mean": {
                            "mean": np.mean(
                                out_values[metric][submetric]["mean"]
                            ).item(),
                            "std": np.std(
                                out_values[metric][submetric]["mean"]
                            ).item(),
                            "median": np.median(
                                out_values[metric][submetric]["mean"]
                            ).item(),
                        },
                        "std": {
                            "mean": np.mean(
                                out_values[metric][submetric]["std"]
                            ).item(),
                            "std": np.std(
                                out_values[metric][submetric]["std"]
                            ).item(),
                            "median": np.median(
                                out_values[metric][submetric]["std"]
                            ).item(),
                        },
                    }
            else:
                for submetric in out_values[metric].keys():
                    out_stats[metric][submetric] = {
                        "mean": np.mean(out_values[metric][submetric]).item(),
                        "std": np.std(out_values[metric][submetric]).item(),
                        "median": np.median(
                            out_values[metric][submetric]
                        ).item(),
                    }

        return out_values, out_stats

    def init_dict(self):
        """
        Initialize an empty dictionary for metrics.

        Returns:
            dict: Dictionary with keys for each metric to compute
        """
        return {key: {} for key in self.tocompute}

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
import pygmtools as pygm

HIGH_COST = 1e3


class Detection:
    """
    A class to handle postionnal information of a recording.

    Attributes:
        doa (np.ndarray): Direction of arrival (DOA) with shape [time, J, 2] where:
            - time: number of time frames
            - J: number of sources
            - 2: azimuth and elevation angles
        vad (np.ndarray): Voice activity detection mask with shape [time, J]
        nos (np.ndarray): Number of active sources per time frame
    """

    def __init__(self, doa: np.ndarray):
        """
        Initialize Detection object with DOA data, and compute VAD and NOS
            - Inactive time frames contain NaN values.

        Args:
            doa (np.ndarray): Direction of arrival data with shape [time, J, 2]
        """

        self.doa = doa
        self.vad = self.get_vad()  # time, J
        self.nos = self.get_nos()

    def get_vad(self):
        """
        Generate VAD mask based on non-NaN values in azimuth.

        Returns:
            np.ndarray: Boolean mask indicating active sources
        """
        return ~np.isnan(self.doa[..., 0])

    def get_nos(self):
        """
        Calculate NOS per time frame.

        Returns:
            np.ndarray: Number of active sources at each time frame
        """
        return self.vad.sum(axis=-1)


class BaseMetric:
    """
    Base class for computing tracking metrics with matching between ground truth and predictions.

    Attributes:
        threshold_gating (float): Maximum allowed distance for matching
        similarity_function (Callable): Function to compute similarity between predictions and ground truth
        similarity_matrix (np.ndarray): Computed similarity matrix
        matches (np.ndarray): Binary matrix indicating matched pairs
        TP (int): Number of true positive matches
    """

    def __init__(self, threshold_gating: float, similarity_function: Callable):
        self.threshold_gating = threshold_gating
        self.similarity_function = similarity_function

        self.similarity_matrix = None
        self.matches = None
        self.TP = None

    def __call__(
        self,
        gt_det: Detection,
        pr_det: Detection,
        similarity_matrix: np.ndarray = None,
        matches: np.ndarray = None,
    ):
        """
        Compute metrics between ground truth and predicted detections.

        Args:
            gt_det (Detection): Ground truth detections
            pr_det (Detection): Predicted detections
            similarity_matrix (np.ndarray, optional): Pre-computed similarity matrix
            matches (np.ndarray, optional): Pre-computed matches

        Returns:
            dict: Computed metrics from the run method
        """
        self.similarity_matrix = similarity_matrix
        self.matches = matches
        if self.similarity_matrix is None or self.matches is None:
            # one to one matching between ground truths and predictions
            self.similarity_matrix = compute_similarity(
                gt_det.doa,
                pr_det.doa,
                threshold_gating=self.threshold_gating,
                similarity_function=self.similarity_function,
            )
            matches_all = pygm.linear_solvers.hungarian(
                s=-self.similarity_matrix
            )
            mask = np.logical_and(
                np.equal(matches_all, 1),
                np.not_equal(self.similarity_matrix, HIGH_COST),
            )
            self.matches = matches_all * mask

        self.TP = self.matches.sum()

        return self.run(gt_det, pr_det)

    def combine_scenes(self, metrics_per_scene: dict, metric_name: str):
        """
        Combine metrics from multiple scenes using weighted average.

        Args:
            metrics_per_scene (dict): Dictionary containing metrics for each scene
            metric_name (str): Name of the metric to combine

        Returns:
            dict: Combined metrics across scenes
        """
        out = {}
        submetrics = self.get_submetrics()
        # a weighted average (number of matches per scene)
        for submetric in submetrics:
            values = np.array(
                [
                    metrics_per_scene[s][metric_name][submetric]
                    for s in metrics_per_scene.keys()
                ]
            )

            weights = np.array(
                [
                    metrics_per_scene[s][metric_name]["TP"]
                    for s in metrics_per_scene.keys()
                ]
            )

            out[submetric] = np.average(values, weights=weights, axis=0).item()

        return out

    def run(self, gt_det: Detection, pr_det: Detection):
        """
        Abstract method to compute specific metrics.

        Args:
            gt_det (Detection): Ground truth detections
            pr_det (Detection): Predicted detections
        """
        raise NotImplementedError

    def get_submetrics(self):
        """
        Abstract method to get list of submetrics.
        """
        raise NotImplementedError


def compute_similarity(
    gt_doa: np.ndarray,
    pr_doa: np.ndarray,
    threshold_gating: float,
    similarity_function: Callable,
):
    """
    Compute similarity matrix between ground truth and predicted DOAs.

    Args:
        gt_doa (np.ndarray): Ground truth DOA with shape [time, Jgt, 2]
        pr_doa (np.ndarray): Predicted DOA with shape [time, Jpr, 2]
        threshold_gating (float): Maximum allowed distance for matching
        similarity_function (Callable): Function to compute similarity between predictions and ground truth

    Returns:
        np.ndarray: Similarity matrix with shape [time, Jgt, Jpr]
    """
    T, Jgt, _ = gt_doa.shape
    T, Jpr, _ = pr_doa.shape

    similarity_matrix = np.empty((T, Jgt, Jpr))

    for jgt in range(Jgt):
        for jpr in range(Jpr):
            gtaz = gt_doa[:, jgt, 0]
            gtel = gt_doa[:, jgt, 1]

            praz = pr_doa[:, jpr, 0]
            prel = pr_doa[:, jpr, 1]

            similarity_matrix[:, jgt, jpr] = similarity_function(
                gtaz, gtel, praz, prel
            )

    similarity_matrix[np.isnan(similarity_matrix)] = HIGH_COST
    similarity_matrix[similarity_matrix > threshold_gating] = HIGH_COST

    return similarity_matrix


def angular_distance(
    az1: np.ndarray, el1: np.ndarray, az2: np.ndarray, el2: np.ndarray
):
    """
    Calculate angular distance between two directions specified by azimuth and elevation.
    Assumes the angles are all in degree

    Returns:
        np.ndarray: Angular distance in degrees
    """
    tmp = np.sin(np.deg2rad(el1)) * np.sin(np.deg2rad(el2)) + np.cos(
        np.deg2rad(el1)
    ) * np.cos(np.deg2rad(el2)) * np.cos(np.deg2rad(az1) - np.deg2rad(az2))
    return np.rad2deg(np.arccos(np.clip(tmp, -1, 1)))


def angdiff(x: np.ndarray, y: np.ndarray):
    """
    Calculate wrapped angular difference between two angles.

    Args:
        x (np.ndarray): First angle in degrees
        y (np.ndarray): Second angle in degrees

    Returns:
        np.ndarray: Angular difference wrapped to [-180, 180] degrees
    """
    return wrap_to_pi(y - x)


def wrap_to_pi(angle):
    # https://stackoverflow.com/questions/2320986/easy-way-to-keeping-angles-between-179-and-180-degrees
    two_pi, pi = 360, 180
    angle = angle % two_pi  # reduce the angle
    angle = (
        (angle + two_pi) % two_pi
    )  # force it to be the positive remainder, so that 0 <= angle < two_pi
    if (
        angle > pi
    ).any():  # force into the minimum absolute value residue class, so that -pi < angle <= pi
        angle -= two_pi
    return angle


def get_fragments(vad):
    """
    Extract continuous fragments from a voice activity detection array.
    A fragment is defined as a continuous sequence of active frames (where vad=1).

    Args:
        vad (np.ndarray): Voice activity detection array with shape [time]

    Returns:
        list: List of [start, end] pairs indicating fragment boundaries
    """

    T = vad.shape[0]

    # Handle special cases where VAD is all ones or zeros
    if (vad == 1).all():
        return [[0, T - 1]]  # One fragment spanning the entire sequence
    elif (vad == 0).all():
        return []  # No fragments when all frames are inactive
    else:
        fragments = []

        # Create array of time indices and separate active/inactive frames
        idx = np.arange(0, T, 1, dtype=int)
        active = idx[vad == 1]  # Indices where VAD is active (1)
        inactive = idx[vad == 0]  # Indices where VAD is inactive (0)

        # Calculate gaps between consecutive active/inactive frames
        # prepend -1 to handle the case where activity starts at index 0
        delays_active = np.diff(active, prepend=[-1])
        delays_inactive = np.diff(inactive, prepend=[-1])

        # Find start and end points of fragments
        # Starts: where gap between active frames > 1
        # Ends: where gap between inactive frames > 1
        idx_starts = np.where(delays_active > 1)[0]
        idx_ends = np.where(delays_inactive > 1)[0]

        # Convert indices to lists
        starts = active[idx_starts].tolist()
        ends = inactive[idx_ends].tolist()

        # Handle edge cases and ensure starts/ends are properly paired
        if len(ends) < len(starts):
            ends.append(idx[-1].item())  # Add end at last frame if missing
        if len(starts) < len(ends):
            starts.insert(0, 0)  # Add start at first frame if missing
            if ends == [T - 1]:
                ends = [T - 2]  # Adjust end if it's the last frame
        if len(starts) == len(ends) and starts[0] > ends[0]:
            starts.insert(
                0, 0
            )  # Add start at beginning if first end comes before first start
            ends.append(idx[-1].item())  # Add end at last frame

        # Remove invalid fragment where start and end are the same
        if ends[-1] == starts[-1]:
            ends = ends[:-1]
            starts = starts[:-1]

        # Create final list of fragments by pairing starts and ends
        fragments = [[start, end] for start, end in zip(starts, ends)]

        return fragments

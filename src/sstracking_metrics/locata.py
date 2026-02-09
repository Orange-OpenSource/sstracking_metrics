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
from scipy.optimize import linear_sum_assignment

from .utils_metrics import HIGH_COST, Detection, angdiff, get_fragments


class LOCATAMetric:
    """
    A class implementing the LOCATA evaluation metrics for SST.

    Original MATLAB code: https://github.com/cevers/sap_locata_eval/blob/master/functions/measures.m
    Attributes:
        az_thresh (float): Azimuth threshold for evaluation
        OSPA_cutoff (float): Cutoff value for OSPA (Optimal SubPattern Assignment) metric
        OSPA_p (list): List of p-values used in OSPA calculation
        frame_len (float): Length of each frame in seconds
    """

    def __init__(self, frame_len: float):
        """
        Initialize the LOCATA metric calculator.

        Args:
            frame_len (float): Length of each frame in seconds
        """
        self.az_thresh = 30
        self.OSPA_cutoff = self.az_thresh
        self.OSPA_p = [1.0, 1.5, 2.0, 5.0]

        self.frame_len = frame_len  # in seconds

    def __call__(
        self,
        gt_det: Detection,
        pr_det: Detection,
    ):
        T = gt_det.doa.shape[0]
        self.timestamps = np.linspace(0, T * self.frame_len, T)

        self.num_trks = pr_det.doa.shape[1]

        self.num_srcs = gt_det.doa.shape[1]
        self.all_src_idx = np.array(list(range(self.num_srcs)))

        # Init:
        self.N_false = np.zeros((T, 1), dtype=int)
        self.N_valid = np.zeros((T, 1), dtype=int)
        self.N_miss = np.zeros((T, self.num_srcs), dtype=int)
        self.N_swap = np.zeros((T, self.num_srcs), dtype=int)
        self.OSPA_dist = np.full((T, len(self.OSPA_p)), np.nan)

        self.assoc_trk_mat = np.full((T, self.num_trks), -1)
        self.prev_nonzero_assign = np.full((self.num_srcs), -1)

        # Filling
        self.fill_values(gt_det, pr_det)

        # Metrics
        metrics = self.get_metrics(gt_det)

        return metrics

    def fill_values(self, gt_det: Detection, pr_det: Detection):
        """
        Fill evaluation matrices by comparing ground truth and predicted detections at each timestamp.
            Processes frame-by-frame to calculate various tracking metrics including:
            - False alarms and valid tracks
            - Missed detections
            - Track swaps
            - OSPA distances

        Args:
            gt_det (Detection): Ground truth detection object
            pr_det (Detection): Predicted detection object
        """

        for t in range(len(self.timestamps)):
            # Number of active sources and number of tracks at this time stamp:
            active_src_idx = []
            for src_idx in range(self.num_srcs):
                if gt_det.vad[t, src_idx] == 1:
                    active_src_idx.append(src_idx)
            num_active_srcs = len(active_src_idx)
            active_src_idx = np.array(active_src_idx)

            active_trk_idx = []
            for trk_idx in range(self.num_trks):
                if pr_det.vad[t, trk_idx] == 1:
                    active_trk_idx.append(trk_idx)
            num_active_trks = len(active_trk_idx)
            active_trk_idx = np.array(active_trk_idx)

            if num_active_srcs == 0 and num_active_trks > 0:
                # No sources
                # Eval num false tracks:
                self.N_false[t] = num_active_trks

                # OSPA
                for p_idx in range(len(self.OSPA_p)):
                    self.OSPA_dist[t, p_idx] = (
                        1
                        / num_active_trks
                        * (
                            self.OSPA_cutoff ** self.OSPA_p[p_idx]
                            * abs(num_active_trks)
                        )
                    ) ** (1 / self.OSPA_p[p_idx])

            elif num_active_srcs > 0 and num_active_trks == 0:
                # No tracks
                # Eval num missing sources
                self.N_miss[t, active_src_idx] = 1

                # OSPA
                for p_idx in range(len(self.OSPA_p)):
                    self.OSPA_dist[t, p_idx] = (
                        1
                        / num_active_srcs
                        * (
                            self.OSPA_cutoff ** self.OSPA_p[p_idx]
                            * abs(num_active_srcs)
                        )
                    ) ** (1 / self.OSPA_p[p_idx])

            elif num_active_srcs > 0 and num_active_trks > 0:
                # Active sources and active tracks

                # @ 1) distance matrix
                dist_mat_az = np.full(
                    (num_active_srcs, num_active_trks), HIGH_COST
                )
                dist_mat_el = np.full(
                    (num_active_srcs, num_active_trks), HIGH_COST
                )
                for trk_idx in range(num_active_trks):
                    for src_idx in range(num_active_srcs):
                        dist_mat_az[src_idx, trk_idx] = abs(
                            angdiff(
                                gt_det.doa[t, active_src_idx[src_idx], 0],
                                pr_det.doa[t, active_trk_idx[trk_idx], 0],
                            )
                        )

                        dist_mat_el[src_idx, trk_idx] = abs(
                            angdiff(
                                gt_det.doa[t, active_src_idx[src_idx], 1],
                                pr_det.doa[t, active_trk_idx[trk_idx], 1],
                            )
                        )

                # Save copy of distance matrix for OSPA before gating:
                dist_mat_4OSPA = dist_mat_az.copy()
                # @ 2) Gating
                dist_mat_az[dist_mat_az > self.az_thresh] = HIGH_COST

                # @ 3) One-to-one mapping between tracks and sources

                match_row, match_col = linear_sum_assignment(
                    dist_mat_az, maximize=False
                )
                assignment = np.full((self.num_srcs,), -1)
                assignment[active_src_idx[match_row]] = active_trk_idx[
                    match_col
                ]

                # OSPA
                for p_idx in range(len(self.OSPA_p)):
                    OSPA_dist_mat = (
                        np.minimum(self.OSPA_cutoff, dist_mat_4OSPA)
                        ** self.OSPA_p[p_idx]
                    )
                    match_row, match_col = linear_sum_assignment(OSPA_dist_mat)
                    cost = OSPA_dist_mat[match_row, match_col].sum()

                    # calculate final distance
                    self.OSPA_dist[t, p_idx] = (
                        1
                        / max(num_active_trks, num_active_srcs)
                        * (
                            self.OSPA_cutoff ** self.OSPA_p[p_idx]
                            * abs(
                                num_active_trks - num_active_srcs
                            )  # card error
                            + cost  # loc error
                        )
                    ) ** (1 / self.OSPA_p[p_idx])

                # # @ Track accuracy
                # valid and false tracks
                unassoc_trks = np.array(
                    [i for i in active_trk_idx if i not in list(assignment)]
                )
                valid_trks = np.array(
                    [i for i in active_trk_idx if i in list(assignment)]
                )

                self.assoc_trk_mat[t, valid_trks] = self.all_src_idx[
                    assignment != -1
                ]
                self.N_false[t] += len(unassoc_trks)
                self.N_valid[t] += len(valid_trks)

                # missing source detections
                unassoc_srcs = np.array(
                    [i for i in active_src_idx if assignment[i] == -1],
                    dtype=int,
                )
                self.N_miss[t, unassoc_srcs] += 1

                # swapped tracks
                if np.any(self.prev_nonzero_assign != assignment):
                    for src_idx in self.all_src_idx:
                        if self.prev_nonzero_assign[src_idx] == -1:
                            # src newborn - ignore
                            continue
                        elif assignment[src_idx] == -1:
                            # source currently either missed or died - ignore
                            continue
                        elif (
                            assignment[src_idx]
                            != self.prev_nonzero_assign[src_idx]
                        ):
                            self.N_swap[t, src_idx] += 1
                # Save assignment for eval of track swaps at next time step:
                for src_idx in self.all_src_idx:
                    if assignment[src_idx] != -1:
                        self.prev_nonzero_assign[src_idx] = assignment[src_idx]

            else:
                continue
        return

    def get_metrics(self, gt_det: Detection):
        """
        Calculate final evaluation metrics based on the filled tracking matrices.

        Args:
            gt_det (Detection): Ground truth detection object

        Returns:
            dict: Dictionary containing:
                - MOTA (float): Multiple Object Tracking Accuracy
                - pd (list): Probability of detection for each source
                - track_latency (list): Average detection delay for each source
                - track_frag_rate (list): Track fragmentation rate
                - track_swap_rate (list): Track swap rate
                - FAR (float): False alarm rate
                - mean_OSPA (float): Mean optimal subpattern assignment distance in degrees
        """

        # MOTA
        MOTA = (1 / len(np.nonzero(self.assoc_trk_mat != -1)[0])) * np.sum(
            self.N_false + self.N_miss + self.N_swap
        )

        # False alarm rate over recording duration
        rec_duration = self.timestamps[-1] - self.timestamps[0]
        FAR = np.sum(self.N_false) / rec_duration

        track_latency = np.zeros(self.num_srcs)
        pd = [[] for _ in range(self.num_srcs)]

        track_swap_rate = np.full(len(self.all_src_idx), np.nan)
        track_frag_rate = np.full(len(self.all_src_idx), np.nan)

        for src_idx in self.all_src_idx:
            fragments = get_fragments(gt_det.vad[:, src_idx])
            VAD_srt_idx = [frag[0] for frag in fragments]
            VAD_end_idx = [frag[1] for frag in fragments]

            # Track latency [1] & Probability of detection
            num_miss_srt = np.zeros(len(VAD_srt_idx), dtype=int)
            miss_duration = np.zeros(len(VAD_srt_idx))
            TSR = np.zeros(len(VAD_srt_idx))
            TFR = np.zeros(len(VAD_srt_idx))

            for period_idx in range(len(VAD_srt_idx)):
                this_miss = self.N_miss[
                    VAD_srt_idx[period_idx] : VAD_end_idx[period_idx], src_idx
                ]
                if np.equal(this_miss, 0).any():
                    # check that the track is detected for at least one sample
                    num_miss_srt[period_idx] = np.nonzero(this_miss == 0)[0][0]
                    miss_duration[period_idx] = self.timestamps[
                        max(0, num_miss_srt[period_idx])
                    ]

                # pd
                pd[src_idx].append(
                    np.sum(1 - this_miss).item() / len(this_miss)
                )

                # TSR
                period_duration = (
                    self.timestamps[VAD_end_idx[period_idx]]
                    - self.timestamps[VAD_srt_idx[period_idx]]
                )
                TSR[period_idx] = (
                    np.sum(
                        self.N_swap[
                            VAD_srt_idx[period_idx] : VAD_end_idx[period_idx],
                            src_idx,
                        ]
                    )
                    / period_duration
                )

                # TFR
                this_period = np.arange(
                    VAD_srt_idx[period_idx], VAD_end_idx[period_idx]
                )
                N_fragment = len(
                    np.nonzero(
                        np.logical_and(
                            np.equal(self.N_miss[this_period[:-1], src_idx], 1),
                            np.equal(self.N_miss[this_period[1:], src_idx], 0),
                        )
                    )[0]
                ) + np.sum(
                    self.N_swap[
                        VAD_srt_idx[period_idx] : VAD_end_idx[period_idx],
                        src_idx,
                    ],
                    axis=0,
                )
                TFR[period_idx] = np.sum(N_fragment) / period_duration

            track_latency[src_idx] = np.mean(miss_duration)
            track_swap_rate[src_idx] = np.mean(TSR)
            track_frag_rate[src_idx] = np.mean(TFR)

        out = {
            "MOTA": float(MOTA),
            "pd": pd,
            "track_latency": track_latency.tolist(),
            "track_frag_rate": track_frag_rate.tolist(),
            "track_swap_rate": track_swap_rate.tolist(),
            "FAR": FAR.item(),
            "mean_OSPA": np.nanmean(self.OSPA_dist).item(),
        }

        return out

    def combine_scenes(self, metrics_per_scene: dict, metric=None):
        """
        Combine metrics from multiple scenes into statistics.
        Calculates mean and std for each metric across all scenes.

        Args:
            metrics_per_scene (dict): Dictionary containing metrics for each scene,
                                      with 'locata' as a key for each scene's metrics

        Returns:
            dict: Combined metrics with mean and standard deviation for:
                - MOTA (%): Multiple Object Tracking Accuracy
                - pd (%): Probability of detection
                - track_latency (s): Track detection latency
                - track_frag_rate (s⁻¹): Track fragmentation rate
                - track_swap_rate: Track swap rate
                - FAR (s⁻¹): False Alarm Rate
                - mean_OSPA (°): Mean OSPA distance
        """

        # in locata seems they just take the mean and the variance
        # Initialize output dictionary

        out = {
            "MOTA": np.array([]),  # %
            "pd": np.array([]),  # %
            "track_latency": np.array([]),  # s
            "track_frag_rate": np.array([]),  # s-1
            "track_swap_rate": np.array([]),
            "FAR": np.array([]),  # s-1
            "mean_OSPA": np.array([]),  # °
        }

        for scene in metrics_per_scene.keys():
            for submetric in metrics_per_scene[scene]["locata"].keys():
                try:
                    toappend = np.array(
                        metrics_per_scene[scene]["locata"][submetric],
                        dtype=np.float32,
                    ).flatten()
                except ValueError:  # list of list doesn't have same len
                    toappend = np.array(
                        [
                            kk
                            for k in metrics_per_scene[scene]["locata"][
                                submetric
                            ]
                            for kk in k
                        ]
                    )
                if np.isnan(toappend).any():
                    toappend = toappend[~np.isnan(toappend)]
                out[submetric] = np.append(out[submetric], toappend)

        for submetric in out.keys():
            l = out[submetric].copy()
            out[submetric] = {
                "mean": np.mean(l).item(),
                "std": np.std(l).item(),
            }

        return out

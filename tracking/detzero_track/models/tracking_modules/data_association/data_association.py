from functools import partial

import torch
import numpy as np

from . import AssigenmentFunc, DistanceFunc


class associate_det_to_tracks:
    def __init__(self, config):
        self.stage = config['stage']['NAME']
        self.class_names = config['class_name']
        self.dist_thresholds = dict()
        for idx, class_n in enumerate(self.class_names):
            self.dist_thresholds[class_n] = config['stage']['FIRST_STAGE']['DIST_THRESHOLD'][idx]

        self.distinguish_class = config['distinguish_class']
        self.assignment_method = partial(AssigenmentFunc[config['assignment_method']])
        self.distance_method = partial(DistanceFunc[config['distance_method']])

        if self.stage == 'two_stage':
            self.point_thresholds = dict()
            self.score_thresholds = dict()
            self.stage_distance_method = dict()
            for idx, class_n in enumerate(self.class_names):
                self.point_thresholds[class_n] = config['stage']['SECOND_STAGE']['POINT_THRESHOLD'][idx]
                self.score_thresholds[class_n] = config['stage']['SECOND_STAGE']['SCORE_THRESHOLD'][idx]
                self.stage_distance_method[class_n] = config['stage']['SECOND_STAGE']['DIST_THRESHOLD'][idx]

    def __call__(self, det_data, track_data):
        if self.stage == 'one_stage':
            return self.one_stage(det_data, track_data, self.dist_thresholds)
        elif self.stage == 'two_stage':
            return self.two_stage(det_data, track_data)

    def one_stage(self, det_data, track_data, dist_thresholds):

        det_box = det_data['boxes_global'][:, :7]
        det_name = det_data['name']
        track_box = track_data['boxes_global'][:, :7]
        track_name = track_data['name']

        affinity_maxtrix = self.distance_method(torch.from_numpy(track_box).float().cuda(), 
                                                torch.from_numpy(det_box).float().cuda())
        track_num, det_num = affinity_maxtrix.shape
        if det_num > 0 and track_num > 0:
            for trk_idx, track_n in enumerate(track_name):
                # affinity of different class is 0
                if self.distinguish_class:
                    diff_mask =  (det_name != track_n)
                    affinity_maxtrix[trk_idx, diff_mask] = 0.

                # filter affinity < threshold into 0
                low_affinity_mask = affinity_maxtrix[trk_idx, :] < dist_thresholds[track_n]
                affinity_maxtrix[trk_idx, low_affinity_mask] = 0.

        cost_matrix = 1.0 - affinity_maxtrix
        matched, tarck_unmatch, det_unmatch = self.assignment_method(cost_matrix)

        return matched, tarck_unmatch, det_unmatch, np.zeros(matched.shape[0], dtype=np.int)

    def two_stage(self, det_data, track_data):
        det_box = det_data['boxes_global']
        det_score = det_data['score']
        num_pts_in_det = det_data['num_points']
        det_name = det_data['name']

        track_box = track_data['boxes_global']
        track_name = track_data['name']

        score_thresholds = np.array([self.score_thresholds[x] for x in det_name])
        point_thresholds = np.array([self.point_thresholds[x] for x in det_name])

        if track_box.shape[0] == 0:
            det_unmatch = np.flatnonzero(np.greater_equal(num_pts_in_det, point_thresholds))
            return np.zeros((0, 2), dtype=np.int), np.arange(0), det_unmatch, np.zeros_like(det_unmatch)

        first_mask = np.greater_equal(det_score, score_thresholds) & np.greater_equal(num_pts_in_det, point_thresholds)
        first_det_index = np.flatnonzero(first_mask)

        first_det_data = {
            'boxes_global': det_box[first_det_index],
            'name': det_name[first_det_index]
        }
        first_matched, tarck_unmatch, det_unmatch, _ = \
            self.one_stage(first_det_data, track_data, self.dist_thresholds)

        first_matched[:, 1] = first_det_index[first_matched[:, 1]]

        frist_det_unmatch_index = first_det_index[det_unmatch]
        second_det_mask = ~first_mask
        second_det_idx = np.flatnonzero(second_det_mask)

        second_trk_mask = np.zeros(track_box.shape[0], dtype=np.bool)
        second_trk_mask[tarck_unmatch] = True
        second_trk_idx = np.flatnonzero(second_trk_mask)

        second_det_data = {
            'boxes_global': det_box[second_det_idx],
            'name': det_name[second_det_idx]
        }
        second_track_data = {
            'boxes_global': track_box[second_trk_idx],
            'name': track_name[second_trk_idx]
        }


        second_matched, tarck_unmatch, det_unmatch, _ = \
            self.one_stage(second_det_data, second_track_data, self.stage_distance_method)

        second_matched[:, 0] = second_trk_idx[second_matched[:, 0]]
        second_matched[:, 1] = second_det_idx[second_matched[:, 1]]

        matched = np.concatenate((first_matched, second_matched), axis=0)
        matched_stage = np.zeros(matched.shape[0], dtype=np.int)
        matched_stage[:first_matched.shape[0]] = 0
        matched_stage[first_matched.shape[0]:] = 1

        tarck_unmatch = second_trk_idx[tarck_unmatch]

        # det_unmatch = np.append(frist_det_unmatch_index, second_det_idx[det_unmatch])
        det_unmatch = frist_det_unmatch_index
        det_unmatch = det_unmatch[np.greater_equal(num_pts_in_det[det_unmatch], point_thresholds[det_unmatch])]

        return matched, tarck_unmatch, det_unmatch, matched_stage

    def only_two_stage(self, det_data, track_data):
        det_box = det_data['boxes_global']
        det_score = det_data['score']
        num_pts_in_det = det_data['num_points']
        det_name = det_data['name']

        track_box = track_data['boxes_global']
        track_name = track_data['name']

        score_thresholds = np.array([self.score_thresholds[x] for x in det_name])
        point_thresholds = np.array([self.point_thresholds[x] for x in det_name])

        if track_box.shape[0] == 0:
            det_unmatch = np.flatnonzero(np.greater_equal(num_pts_in_det, point_thresholds))
            return np.zeros((0, 2), dtype=np.int), np.arange(0), det_unmatch

        first_mask = np.greater_equal(det_score, score_thresholds) & np.greater_equal(num_pts_in_det, point_thresholds)
        second_det_mask = ~first_mask
        second_det_idx = np.flatnonzero(second_det_mask)
        second_det_data = {
            'boxes_global': det_box[second_det_idx],
            'name': det_name[second_det_idx]
        }

        second_matched, tarck_unmatch, det_unmatch, _ = \
            self.one_stage(second_det_data, track_data, self.stage_distance_method)
        second_matched[:, 1] = second_det_idx[second_matched[:, 1]]

        return second_matched, tarck_unmatch, second_det_idx[det_unmatch]

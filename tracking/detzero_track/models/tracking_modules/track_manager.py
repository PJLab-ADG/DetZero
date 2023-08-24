import copy
from easydict import EasyDict
from functools import partial
from collections import defaultdict

import torch
import numpy as np

from . import data_association, kalman_filter


class TrackManager():
    def __init__(self, model_cfg, init_track_id=0):
        self.model_cfg = model_cfg
        self.init_track_id = init_track_id

        self.module_topology = [
            'filter', 'track_age', 'data_association', 
            'track_merge', 'reverse_tracking'
        ]
        self.modules_dicts = self.build_modules()

    def build_modules(self):
        modules_dicts = {}
        for module_name in self.module_topology:
            modules_dicts = getattr(self, 'build_%s' % module_name)(
                modules_dicts=modules_dicts
            )
        return modules_dicts

    def build_filter(self, modules_dicts):
        if self.model_cfg.get('FILTER', None) is None:
            return modules_dicts

        filter_cfg = copy.deepcopy(self.model_cfg.FILTER)
        filter_cfg = EasyDict({key.lower(): value for key, value in filter_cfg.items()})
        filter_module = partial(kalman_filter.__FILTER__[filter_cfg.name], **filter_cfg)
        modules_dicts['filter_module'] = filter_module
        modules_dicts['filter_config'] = filter_cfg
        return modules_dicts

    def build_track_age(self, modules_dicts):
        if self.model_cfg.get('TRACK_AGE', None) is None:
            return modules_dicts

        age_cfg = copy.deepcopy(self.model_cfg.TRACK_AGE)
        age_cfg = EasyDict({key.lower(): value for key, value in age_cfg.items()})
        modules_dicts['track_age_config'] = age_cfg
        return modules_dicts

    def build_data_association(self, modules_dicts):
        if self.model_cfg.get('DATA_ASSOCIATION', None) is None:
            return modules_dicts

        assoc_cfg = copy.deepcopy(self.model_cfg.DATA_ASSOCIATION)
        assoc_cfg = EasyDict({key.lower(): value for key, value in assoc_cfg.items()})
        assoc_module = data_association.associate_det_to_tracks(config=assoc_cfg)
        modules_dicts['data_association_module'] = assoc_module
        modules_dicts['data_association_config'] = assoc_cfg
        return modules_dicts

    def build_track_merge(self, modules_dicts):
        if self.model_cfg.get('TRACK_MERGE', None) is None:
            return modules_dicts

        merge_cfg = copy.deepcopy(self.model_cfg.TRACK_MERGE)
        merge_cfg = EasyDict({key.lower(): value for key, value in merge_cfg.items()})
        if merge_cfg.enable:
            merge_thresh = dict()
            for idx, class_n in enumerate(merge_cfg['class_name']):
                merge_thresh[class_n] = merge_cfg['class_threshold'][idx]
            merge_cfg['class_threshold'] = merge_thresh
        modules_dicts['track_merge_config'] = merge_cfg
        return modules_dicts

    def build_reverse_tracking(self, modules_dicts):
        if self.model_cfg.get('REVERSE_TRACKING', None) is None:
            return modules_dicts

        reverse_cfg = copy.deepcopy(self.model_cfg.REVERSE_TRACKING)
        reverse_cfg = EasyDict({key.lower(): value for key, value in reverse_cfg.items()})
        modules_dicts['reverse_tracking_config'] = reverse_cfg
        return modules_dicts

    def forward(self, data_dict):
        frame_list = sorted(list(data_dict.keys()), key=int)
        tracks = list()
        tk_result = dict()
        tk_id_cnt = self.init_track_id
        
        for _, frm_id in enumerate(frame_list):
            # execute tracking module frame by frame
            frm_tk_data, tracks, tk_id_cnt = self.online_track_module(
                frm_id, data_dict[frm_id], tracks, tk_id_cnt
            )
            # transfer the tracking result from frame-level
            # to object-level structure 
            for key, val in frm_tk_data.items():
                if key not in tk_result.keys():
                    tk_result[key] = defaultdict(list)
                for sub_key, sub_val in val.items():
                    tk_result[key][sub_key].append(sub_val)
                tk_result[key]['pose'].append(data_dict[frm_id]['pose'])

        for tk_id in tk_result.keys():
            for key in tk_result[tk_id].keys():
                tk_result[tk_id][key] = np.array(tk_result[tk_id][key])

        # execute the reverse-order tracking stage
        if self.modules_dicts['reverse_tracking_config'].enable:
            frm_tracks = dict()
            reverse_tracks = list()
            keys = ['boxes_global', 'name', 'score', 'sample_idx',
                    'hit', 'num_points', 'obj_ids']

            # transfer the tracking result to frame-level strutcure
            for tk_id in tk_result.keys():
                sample_idx = tk_result[tk_id]['sample_idx']
                for i, sa_idx in enumerate(sample_idx):
                    if sa_idx not in frm_tracks.keys():
                        frm_tracks[sa_idx] = defaultdict(list)
                    frm_tracks[sa_idx]['start'].append(1 if i == 0 else 0)
                    for key in keys:
                        frm_tracks[sa_idx][key].append(tk_result[tk_id][key][i])

            for key, items in frm_tracks.items():
                for k, v in items.items():
                    items[k] = np.array(v)

            for idx, frm_id in enumerate(frame_list[::-1]):
                frm_tk_data, reverse_tracks = self.reverse_tracking_module(
                    frm_id, data_dict[frm_id], frm_tracks[frm_id], reverse_tracks
                )
                for key, val in frm_tk_data.items():
                    for sub_key, sub_val in val.items():
                        tk_result[key][sub_key] = np.insert(
                            tk_result[key][sub_key], 0, sub_val, axis=0
                        )
                    tk_result[key]['pose'] = np.insert(
                            tk_result[key]['pose'], 0, data_dict[frm_id]['pose'], axis=0
                        )

        return tk_result

    def predict_tracks(self, frm_id, tracks):
        tk_boxes = np.zeros((len(tracks), 9), dtype=np.float32)
        tk_name = list()
        tk_score = list()

        for i, tk in enumerate(tracks):
            tk_boxes[i] = tk.predict(frm_id)[:9]
            tk_name.append(tk.name)
            tk_score.append(tk.score)

        tk_data = {
            'boxes_global': np.array(tk_boxes),
            'name': np.array(tk_name),
            'score': np.array(tk_score),
        }
        return tk_data

    def online_track_module(self, frame_id, det_data, tracks, track_id_count):
        track_data = self.predict_tracks(frame_id, tracks)

        da_stage = (self.modules_dicts['data_association_config'].stage.NAME == 'one_stage')
        if not da_stage and 'num_points' not in det_data.keys():
            det_data['num_points'] = np.zeros_like(det_data['score'])

        matched, track_unmatch, det_unmatch, matched_stage = \
            self.modules_dicts['data_association_module'](det_data, track_data)

        det_boxes = det_data['boxes_global']
        det_name = det_data['name']
        for match_idx, match in enumerate(matched):
            tk_idx, det_idx = match
            tracks[tk_idx].update(
                det_boxes[det_idx],
                det_name[det_idx],
                det_data['score'][det_idx],
                det_data['num_points'][det_idx] if not da_stage else 0,
                two_stage=matched_stage[match_idx]
            )

        for _, det_idx in enumerate(det_unmatch):
            tracks.append(self.modules_dicts['filter_module'](
                bbox=det_boxes[det_idx],
                name=det_name[det_idx], 
                score=det_data['score'][det_idx],
                frame_id=frame_id,
                track_id=track_id_count, 
                num_points=det_data['num_points'][det_idx] if not da_stage else 0,
            ))
            track_id_count += 1

        if self.modules_dicts['track_merge_config']['enable']:
            tracks = self.overlap_track_merge(tracks)

        track_output_data = dict()
        filter_cfg = self.modules_dicts['filter_config']
        track_age_cfg = self.modules_dicts['track_age_config']
        for track in tracks:
            if filter_cfg['name'] == "AB3DMOT":
                if (track.hits >= track_age_cfg.birth_age or int(frame_id) < track_age_cfg.birth_age) and \
                    track.miss < track_age_cfg.death_age:
                    track_output_data.update(copy.deepcopy(track.info()))
            else:
                track_output_data.update(copy.deepcopy(track.info()))

        tracks_idx = len(tracks)
        for track in reversed(tracks):
            tracks_idx -= 1
            if track.miss >= track_age_cfg.death_age and \
                (track_age_cfg.death_age != -1):
                tracks.pop(tracks_idx)

        return track_output_data, tracks, track_id_count

    def reverse_tracking_module(self, frame_id, det_data, trk_data, tracks):
        track_data = self.predict_tracks(frame_id, tracks)
        trk_mask = ~ trk_data['start'].astype(np.bool)

        for key in track_data.keys():
            track_data[key] = np.concatenate((
                track_data[key], trk_data[key][trk_mask]), axis=0)

        da_stage = (self.modules_dicts['data_association_config'].stage.NAME == 'one_stage')
        if not da_stage and 'num_points' not in det_data.keys():
            det_data['num_points'] = np.zeros_like(det_data['score'])

        matched, track_unmatch, det_unmatch = \
            self.modules_dicts['data_association_module'].only_two_stage(det_data, track_data)

        det_boxes = det_data['boxes_global'][:, :9]
        det_name = det_data['name']
        for match_idx, match in enumerate(matched):
            trk_idx, det_idx = match
            if trk_idx >= len(tracks): continue
            tracks[trk_idx].update(det_boxes[det_idx], det_name[det_idx], det_data['score'][det_idx],
                                      det_data['num_points'][det_idx] if not da_stage else 0, two_stage=True)

        if self.modules_dicts['track_merge_config']['enable']:
            tracks = self.overlap_track_merge(tracks)

        track_output_data = dict()
        for track in tracks:
            track_output_data.update(copy.deepcopy(track.info()))

        for obj_idx, obj_id in enumerate(trk_data['obj_ids']):
            if trk_data['start'][obj_idx] == 0:
                continue
            tracks.append(self.modules_dicts['filter_module'](
                bbox=trk_data['boxes_global'][obj_idx][:7], 
                name=trk_data['name'][obj_idx],
                score=trk_data['score'][obj_idx],
                frame_id=frame_id,
                track_id=trk_data['obj_ids'][obj_idx],
                num_points=trk_data['num_points'][obj_idx],
                delta_t=-0.1))
        
        return track_output_data, tracks

    def overlap_track_merge(self, tracks):
        tk_boxes = np.zeros((len(tracks), 7), dtype=np.float32)
        tk_age = np.zeros(len(tracks), dtype=np.int)
        tk_area = np.zeros(len(tracks), dtype=np.float32)
        tk_name = list()

        for i, track in enumerate(tracks):
            tk_boxes[i] = track.bbox[:7]
            tk_age[i] = track.track_id
            tk_area[i] = tk_boxes[i][3] * tk_boxes[i][4]
            tk_name.append(track.name)
        tk_name = np.array(tk_name)

        tk_boxes_cuda = torch.from_numpy(tk_boxes).float().cuda()
        bev_overlap_areas = data_association.bev_overlap_gpu(tk_boxes_cuda, tk_boxes_cuda)
        tk_num, _ = bev_overlap_areas.shape
        if tk_num > 0:
            for i, name in enumerate(tk_name):
                diff_mask = (tk_name != name)
                bev_overlap_areas[i, diff_mask] = 0.

        keep_index_set = set()
        deprecate_index_set = set()
        for i in range(len(tk_boxes)):
            if i in deprecate_index_set or i in keep_index_set: 
                continue

            areas = bev_overlap_areas[i]
            name = tk_name[i]
            overlap_thresh = self.modules_dicts['track_merge_config']['class_threshold'][name]

            overlap_ratio = areas / (tk_area[i]+1e-9)
            overlap_mask = overlap_ratio >= overlap_thresh
            overlap_index = np.flatnonzero(overlap_mask)

            overlap_ages = tk_age[overlap_index]
            sort_idx = np.argsort(overlap_ages)
            best_idx = overlap_index[sort_idx[0]]
            if best_idx not in deprecate_index_set:
                keep_index_set.add(best_idx)
                bev_overlap_areas[:, overlap_index] = 0.
                deprecate_index_set.update(overlap_index[sort_idx[1:]])

        keep_index_list = sorted(list(keep_index_set))
        deprecate_index_list = sorted(list(deprecate_index_set))
        for idx in reversed(deprecate_index_list):
            tracks.pop(idx)

        return tracks

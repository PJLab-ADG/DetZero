import copy
from functools import partial

import torch
import numpy as np

from .data_association import bev_overlap_gpu


class PostProcessor():
    def __init__(self, processor_configs):
        self.post_process_queue = []
        for cur_cfg in processor_configs.CONFIG_LIST:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.post_process_queue.append(cur_processor)

    def forward(self, data_dict):
        for cur_processor in self.post_process_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict

    def empty_track_delete(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.empty_track_delete, config=config)

        remove_tk_id = list()
        for tk_id, tk_data in data_dict.items():
            track_history = len(tk_data['hit'])
            hit_count = np.sum(tk_data['hit'] > 0)

            if hit_count < config.LEAST_AGE:
                remove_tk_id.append(tk_id)
            else:
                if hit_count != track_history:
                    remove_indexs = list()
                    for idx in range(track_history):
                        if tk_data['hit'][idx] >= 1: break
                        else: remove_indexs.append(idx)
                    for idx in reversed(range(track_history)):
                        if tk_data['hit'][idx] >= 1: break
                        else: remove_indexs.append(idx)

                    # remove_indexs=sorted(remove_indexs, reverse=True)
                    if config.END_REMOVE:
                        for key in tk_data.keys():
                            tk_data[key] = np.delete(
                                tk_data[key], remove_indexs, axis=0
                            )

        for tk_id in remove_tk_id:
            data_dict.pop(tk_id)
        return data_dict

    def velocity_optimize(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.velocity_optimize, config=config)

        header_len = config.HEADER_LENGTH
        for tk_id, tk_data in data_dict.items():
            track_len = len(tk_data['boxes_global'])

            if track_len < 2: continue
            process_len = header_len if track_len > header_len else track_len-1
            for idx in range(process_len):
                speed = (tk_data['boxes_global'][idx+1, :2] - tk_data['boxes_global'][idx, :2])*10.
                data_dict[tk_id]['boxes_global'][idx, 7:9] = copy.deepcopy(speed)
            if process_len == track_len:
                data_dict[tk_id]['boxes_global'][-1, 7:9] = copy.deepcopy(tk_data['boxes_global'][-2, 7:9])
        return data_dict

    def motion_classify(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.motion_classify, config=config)

        for tk_id, tk_data in data_dict.items():
            hit_index = np.flatnonzero(tk_data['hit'] == 1)
            track_len = len(hit_index)
            if track_len < 2:
                data_dict[tk_id]['state'] = "static"
            else:
                track_box = torch.from_numpy(tk_data['boxes_global'][hit_index, :7]).float().cuda()
                bev_iou_mat = bev_overlap_gpu(track_box, track_box)
                if np.any(bev_iou_mat  <= 1e-4):
                    data_dict[tk_id]['state'] = "dynamic"
                else:
                    data_dict[tk_id]['state'] = "static"

        return data_dict

    def static_drift_eliminate(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.static_drift_eliminate, config=config)

        for tk_id, tk_data in data_dict.items():
            if tk_data['state'] == 'static' and tk_data['name'][0] == 'Vehicle':
                hit_idxs = np.flatnonzero(tk_data['hit'] == 1)
                temp_max_score_idx = np.argsort(tk_data['score'][hit_idxs])[-1]
                hit_max_score_idx = hit_idxs[temp_max_score_idx]

                track_history = len(tk_data['hit'])
                for idx in reversed(range(track_history)):
                    if tk_data['hit'][idx] >= 1: break
                    else:
                        data_dict[tk_id]['boxes_global'][idx] = copy.deepcopy(
                            tk_data['boxes_global'][hit_max_score_idx])
        return data_dict

    def box_size_update(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.box_size_update, config=config)

        for tk_id, tk_data in data_dict.items():
            scores = tk_data['score']
            boxes = tk_data['boxes_global']

            if config.METHOD == 'max_score_box':
                max_score_indexs = np.where(scores == np.max(scores))[0]
                max_boxes = np.zeros(3, dtype=np.float32)
                for idx in max_score_indexs:
                    max_boxes += boxes[idx, 3:6]
                max_boxes = max_boxes / (len(max_score_indexs))
                data_dict[tk_id]['boxes_global'][:, 3:6] = max_boxes

            elif config.METHOD == 'score_weigthed_box':
                weighted_boxes = np.zeros(3, dtype=np.float32)
                for idx in range(len(scores)):
                    weighted_boxes += scores[idx] * boxes[idx, 3:6]
                weighted_boxes = weighted_boxes/np.sum(scores)
                data_dict[tk_id]['boxes_global'][:, 3:6] = weighted_boxes

            elif config.METHOD == 'largest_box':
                area = boxes[:, 3] * boxes[:, 4] * boxes[:, 5]
                largest_idx = np.argsort(area)[-1]
                largest_boxes = copy.deepcopy(boxes[largest_idx, 3:6])
                data_dict[tk_id]['boxes_global'][:, 3:6] = largest_boxes

        return data_dict

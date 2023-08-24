import os
import copy

import torch
import numpy as np
from functools import partial

from detzero_utils.ops.roiaware_pool3d import roiaware_pool3d_utils

from detzero_track.utils.transform_utils import yaw_filter, transform_boxes3d
from detzero_track.models.tracking_modules.data_association import bev_overlap_gpu


class DataProcessor(object):
    """
    """
    def __init__(self, processor_configs, lidar_path=None):
        self.data_processor_queue = []
        self.lidar_path = lidar_path
        self.ignore_key_list = ['sequence_name', 'timestamp', 'pose', 'frame_id']
        
        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)
    
    def forward(self, data_dict):
        processed_data = dict()
        remove_data = dict()
        sample_idx_list = sorted(list(data_dict.keys()), key=int)
        for sample_idx in sample_idx_list:
            curr_data = data_dict[sample_idx]

            for cur_processor in self.data_processor_queue:
                curr_data = cur_processor(data_dict=curr_data)
                if isinstance(curr_data, tuple):
                    remove_data[sample_idx] = curr_data[1]
                    curr_data = curr_data[0] 
            processed_data[sample_idx] = curr_data

        return processed_data, remove_data

    def heading_process(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.heading_process, config=config)

        if data_dict.get('boxes_lidar', None) is not None:
            heading = data_dict["boxes_lidar"][:, 6]
            data_dict["boxes_lidar"][:, 6] = yaw_filter(heading)
        return data_dict

    def points_in_box(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.points_in_box, config=config)

        if data_dict.get('boxes_lidar', None) is not None:
            seq_name = data_dict['sequence_name']
            if 'segment-' not in seq_name:
                seq_name = 'segment-' + seq_name 

            frame_id = ('0000' + str(data_dict['frame_id']))[-4:] +'.npy'
            lidar_path = os.path.join(self.lidar_path, seq_name, frame_id)
            points = np.load(lidar_path)

            num_points_box = roiaware_pool3d_utils.points_in_boxes_num_gpu(
                torch.from_numpy(points[:, :3]).float().cuda().unsqueeze(0), 
                torch.from_numpy(data_dict['boxes_lidar'][:, :7]).float().cuda().unsqueeze(0)
            )
            num_points_box = num_points_box.squeeze(0).cpu().numpy()
            data_dict['num_points'] = num_points_box

        return data_dict

    def low_confidence_box_filter(self, data_dict=None, config=None, threshold=0.):
        if data_dict is None:
            return partial(self.low_confidence_box_filter, config=config)

        if data_dict.get('score', None) is not None:
            score_mask = data_dict["score"] >= config.THRESHOLD
            for key in list(data_dict.keys()):
                if key in self.ignore_key_list: continue
                data_dict[key] = data_dict[key][score_mask]

        return data_dict

    def transform_to_global(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.transform_to_global, config=config)
        
        if data_dict.get('pose', None) is not None:
            pose = data_dict["pose"]
            boxes_lidar = data_dict["boxes_lidar"]

            boxes_global = transform_boxes3d(boxes_lidar, pose)
            data_dict["boxes_global"] = boxes_global
        return data_dict

    def overlap_box_filter(self, data_dict=None, config=None):
        """
        Function:
            process the ovelap boxes from detection data name_index
        Args:
            data: detection data
            method: the method of output the box from a set of overlaped bboxes
            class_threshold: the threshold of whether overlap or not
        Returns:
            data: already preprocessed detection data
        """
        if data_dict is None:
            return partial(self.overlap_box_filter, config=config)
        
        remove_data_dict = {}
        if data_dict.get('boxes_lidar', None) is not None:
            boxes_lidar = data_dict["boxes_lidar"]
            names = data_dict["name"]
            scores = data_dict["score"]
            if len(names) == 0:
                return data_dict

            boxes_lidar_gpu = torch.from_numpy(boxes_lidar[:, :7]).float().cuda()
            bev_overlap_area = bev_overlap_gpu(boxes_lidar_gpu, boxes_lidar_gpu)

            keep_index_set = set()
            for box_idx in range(len(boxes_lidar)):
                if (box_idx in keep_index_set): continue

                box = boxes_lidar[box_idx]
                name = names[box_idx]
                threshold = config.CLASS_THRESHOLD[name]

                overlap_rate = bev_overlap_area[box_idx]/(box[3]*box[4])
                overlap_index = np.flatnonzero(overlap_rate >= threshold)

                overlap_score = scores[overlap_index]
                sort_idx = np.argsort(overlap_score)
                best_idx = overlap_index[sort_idx[-1]]
                keep_index_set.add(best_idx)

                if config.METHOD == "weigthed_size":
                    total_score = np.sum(overlap_score)
                    overlap_score = np.repeat(overlap_score.reshape(-1, 1), 3, axis=1)
                    weight_size = np.sum(boxes_lidar[overlap_index][:, 3:6] * 
                                         overlap_score, axis=0) / (total_score+1e-9)
                    data_dict['boxes_lidar'][best_idx, 3:6] = weight_size

                elif config.METHOD == "merge_box":
                    total_score = np.sum(overlap_score)
                    overlap_score = np.repeat(overlap_score.reshape(-1, 1), 6, axis=1)
                    merge_box = np.sum(boxes_lidar[overlap_index][:, 0:6] * 
                                       overlap_score, axis=0) / (total_score+1e-9)
                    data_dict['boxes_lidar'][best_idx, 0:6] = merge_box

            keep_index_list = sorted(list(keep_index_set))
            total_index_list = np.arange(len(boxes_lidar))
            remove_index_list = np.setdiff1d(total_index_list, keep_index_list)
            for key in data_dict.keys():
                if key in self.ignore_key_list: 
                    remove_data_dict[key] = copy.deepcopy(data_dict[key])
                else:
                    remove_data_dict[key] = copy.deepcopy(data_dict[key][remove_index_list])
                    data_dict[key] = data_dict[key][keep_index_list]

        return data_dict, remove_data_dict


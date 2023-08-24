import copy
import pickle
from abc import abstractmethod
from collections import defaultdict

import numpy as np
import torch

from detzero_utils import box_utils, common_utils

from .processor.data_processor import DataProcessor
from .processor.point_feature_encoder import PointFeatureEncoder


class DatasetTemplate(torch.utils.data.Dataset):
    """
    The base class of datasets. 
    The design of this class is to decouple all file-related operations and only keeps the computing logics.
    Please do not instantiate this class directly.
    
    """
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.training = training
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.logger = logger

        self.tta = False if self.training else getattr(self.dataset_cfg, "TTA", False)
        self.sweep_count = self.dataset_cfg.get('SWEEP_COUNT', None)
        self.sampled_interval = self.dataset_cfg.SAMPLED_INTERVAL[self.mode]\
            if 'SAMPLED_INTERVAL' in self.dataset_cfg else None

        self.point_cloud_range = np.array(self.dataset_cfg.POINT_CLOUD_RANGE, dtype=np.float32)
        self.point_feature_encoder = PointFeatureEncoder(
            self.dataset_cfg.POINT_FEATURE_ENCODING,
            point_cloud_range=self.point_cloud_range
        )
        self.data_augmentor = self.init_data_augmentor()
        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            point_cloud_range=self.point_cloud_range,
            training=self.training,
            num_point_features=self.point_feature_encoder.num_point_features
        )
        self.test_time_augmentor = self.init_tta()

        self.grid_size = self.data_processor.grid_size
        self.voxel_size = self.data_processor.voxel_size
        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.infos = []

    @property
    def mode(self) -> str: 
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs
        return len(self.infos)

    @abstractmethod
    def get_infos_and_points(self, idx_list):
        """
        Produce a list of infos and points by a idx_list of self.infos. 
        This is an abstract method, please overwrite in subclasses.
        """

    @abstractmethod
    def init_infos(self):
        """
        Load all infos into self.infos. 
        This is an abstract method, please overwrite in subclasses.
        """
        raise NotImplementedError

    def init_data_augmentor(self):
        """
        Load the data augmentor.
        Please overwrite this function if you want to use data augmentor
        """
        return None

    def init_tta(self):
        """
        Load the test times augmentor.
        Please overwrite this function if you want to use data augmentor
        """
        return None

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)
        
        current_info = copy.deepcopy(self.infos[index])
        target_idx_list = self.get_sweep_idxs(current_info, self.sweep_count, index)
        target_infos, points = self.get_infos_and_points(target_idx_list)
        points = self.merge_sweeps(current_info, target_infos, points)

        input_dict = {
            'points': points,
            'frame_id': current_info['sample_idx'],
            'pose': current_info['pose'],
            'sequence_name': current_info['sequence_name'],
        }
            
        if 'annos' in current_info:
            annos = current_info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']
                
            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                # 'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    @staticmethod
    def get_sweep_idxs(current_info, sweep_count=[0, 0], current_idx=0):
        """
        TODO: Docstrings

        """

        assert type(sweep_count) is list and len(sweep_count) == 2,\
            "Please give the upper and lower range of frames you want to process!"

        current_sample_idx = current_info["sample_idx"]
        current_seq_len = current_info["sequence_len"]

        target_sweep_list = np.array(list(range(sweep_count[0], sweep_count[1]+1)))
        target_sample_list = current_sample_idx + target_sweep_list
        # set the high and low thresh to extract multi frames in current sequence
        target_sample_list = [i if i >= 0 else 0 for i in target_sample_list]
        target_sample_list = [i if i < current_seq_len else current_seq_len-1 for i in target_sample_list]
        # get the index of target frames in the waymo info list
        target_idx_res = np.array(target_sample_list) - current_sample_idx
        target_idx_list = current_idx + target_idx_res

        return target_idx_list

    @staticmethod
    def merge_sweeps(info, target_infos, points):
        """
        TODO: Docstrings

        """

        current_pose = info["pose"]
        current_time = info["time_stamp"]
        
        point_clouds = []
        for i in range(len(target_infos)):
            target_info = target_infos[i]
            current_points = points[i]

            current_points, NLZ_flag = current_points[:, 0:5], current_points[:, 5]
            current_points = current_points[NLZ_flag == -1]
            current_points[:, 3] = np.tanh(current_points[:, 3])    # process the intensity into [-1, 1]

            transform_mat = np.linalg.inv(current_pose) @ target_info['pose']
            delta_time = int(target_info['time_stamp']) - int(current_time)
            current_points[:, :3] = np.concatenate([current_points[:, :3], np.ones((current_points.shape[0], 1))],
                                                   axis=1) @ transform_mat[:3, :].T
            time_offset = float(delta_time) / 1000000. * np.ones((current_points.shape[0], 1))
            current_points = np.concatenate([current_points, time_offset], axis=1)
            point_clouds.append(current_points)

        point_clouds = np.concatenate(point_clouds, axis=0)

        return point_clouds

    def prepare_data(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
            data_dict:
                frame_id: string
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                use_lead_xyz: bool
                voxels: optional (num_voxels, max_points_per_voxel, 3 + C)
                voxel_coords: optional (num_voxels, 3)
                voxel_num_points: optional (num_voxels)
                ...
        """

        if self.training:
            assert 'gt_boxes' in data_dict, 'gt_boxes should be provided for training'
            gt_boxes_mask = np.array([n in self.class_names for n in data_dict['gt_names']], dtype=np.bool_)

            data_dict = self.data_augmentor.forward(
                data_dict={
                    **data_dict,
                    'gt_boxes_mask': gt_boxes_mask
                }
            )

        if data_dict.get('gt_boxes', None) is not None:
            selected = common_utils.keep_arrays_by_name(data_dict['gt_names'], self.class_names)
            data_dict['gt_boxes'] = data_dict['gt_boxes'][selected]
            data_dict['gt_names'] = data_dict['gt_names'][selected]
            gt_classes = np.array([self.class_names.index(n) + 1 for n in data_dict['gt_names']], dtype=np.int32)
            gt_boxes = np.concatenate((data_dict['gt_boxes'], gt_classes.reshape(-1, 1).astype(np.float32)), axis=1)
            data_dict['gt_boxes'] = gt_boxes

        data_dict = self.point_feature_encoder.forward(data_dict)

        if self.tta:
            data_dict = self.test_time_augmentor.forward(
                data_dict={**data_dict}
            )
            for key, val in data_dict.items():
                data_dict[key] = self.data_processor.forward(data_dict=val)
                data_dict[key].pop('gt_names', None)
        else:
            data_dict = self.data_processor.forward(
                data_dict=data_dict
            )

        if self.training and len(data_dict['gt_boxes']) == 0:
            new_index = np.random.randint(self.__len__())
            return self.__getitem__(new_index)

        data_dict.pop('gt_names', None)
        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        batch_size = len(batch_list)
        tta = 'tta_original' in batch_list[0]

        for i, cur_sample in enumerate(batch_list):
            if tta:
                tta_ops = cur_sample.keys()
                data_dict['tta_ops'] = [tta_cfg for tta_cfg in tta_ops]
                for key in cur_sample["tta_original"]:
                    if key in ['points', 'voxels', 'voxel_num_points', 'voxel_coords']:
                        for tta_cfg in tta_ops:
                            data_dict[key].append(cur_sample[tta_cfg][key])
                    else:
                        data_dict[key].append(cur_sample["tta_original"][key])
            else:
                for key, val in cur_sample.items():
                    data_dict[key].append(val)

        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['voxels', 'voxel_num_points']:
                    ret[key] = np.concatenate(val, axis=0)
                elif key in ['points', 'voxel_coords']:
                    coors = []
                    for i, coor in enumerate(val):
                        coor_pad = np.pad(coor, ((0, 0), (1, 0)), mode='constant', constant_values=i)
                        coors.append(coor_pad)
                    ret[key] = np.concatenate(coors, axis=0)
                elif key in ['gt_boxes']:
                    max_gt = max([len(x) for x in val])
                    batch_gt_boxes3d = np.zeros((batch_size, max_gt, val[0].shape[-1]), dtype=np.float32)
                    for k in range(batch_size):
                        batch_gt_boxes3d[k, :val[k].__len__(), :] = val[k]
                    ret[key] = batch_gt_boxes3d
                else:
                    ret[key] = np.stack(val, axis=0)
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError
        ret['batch_size'] = batch_size if not tta else int(batch_size * len(tta_ops))

        return ret

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        This is the Waymo version. Further refactor for custom dataset later.


        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:
            annos: list of detection results
        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 9])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict
            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['sequence_name'] = batch_dict['sequence_name'][index]
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['pose'] = batch_dict['pose'][index]
            annos.append(single_pred_dict)

        return annos

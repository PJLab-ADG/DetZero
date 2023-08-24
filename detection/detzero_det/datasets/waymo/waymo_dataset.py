import os
import pickle
import copy

import numpy as np

from detzero_utils import common_utils
from detzero_utils.ops.roiaware_pool3d import roiaware_pool3d_utils

from detzero_det.datasets.dataset import DatasetTemplate
from detzero_det.datasets.augmentor.data_augmentor import DataAugmentor
from detzero_det.datasets.augmentor.test_time_augmentor import TestTimeAugmentor


class WaymoDetectionDataset(DatasetTemplate):
    """
    The Dataset class for Training on Waymo (from File System)
    """

    def __init__(self, dataset_cfg, class_names, root_path, training=True, logger=None) -> None:
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        self.data_path = self.root_path + '/' + dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.init_infos()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names,
            training=self.training, root_path=self.root_path,
            logger=self.logger
        )
        self.split = split
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split+'.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.init_infos()

    def init_data_augmentor(self):
        data_augmentor = DataAugmentor(
            self.root_path,
            self.dataset_cfg.DATA_AUGMENTOR,
            self.class_names,
            logger=self.logger
        ) if self.training else None
        return data_augmentor

    def init_tta(self):
        test_time_augmentor = TestTimeAugmentor(
            self.dataset_cfg.TEST_TIME_AUGMENTOR,
            logger=self.logger
        ) if self.tta else None
        return test_time_augmentor

    def init_infos(self):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []
        num_skipped_infos = 0
        
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = os.path.join(self.data_path, sequence_name, ('%s.pkl' % sequence_name))
            info_path = self.check_sequence_name_with_all_version(info_path)

            if not os.path.exists(info_path):
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)

            waymo_infos.extend(infos)
        
        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % len(waymo_infos))
        if self.dataset_cfg.SAMPLED_INTERVAL[self.mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[self.mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    def check_sequence_name_with_all_version(self, seq_file):
        if '_with_camera_labels' not in seq_file and not os.path.exists(seq_file):
            seq_file = seq_file[:-9] + '_with_camera_labels.tfrecord'
        if '_with_camera_labels' in seq_file and not os.path.exists(seq_file):
            seq_file = seq_file.replace('_with_camera_labels', '')

        return seq_file

    def get_infos_and_points(self, idx_list):
        infos, points = [], []
        for i in idx_list:
            lidar_path = self.infos[i]['lidar_path']
            current_point = np.load(lidar_path)

            infos.append(self.infos[i])
            points.append(current_point)

        return infos, points

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval_detection import WaymoDetectionMetricsEstimator
            eval = WaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)

        return ap_result_str, ap_dict

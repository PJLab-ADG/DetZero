import os
import pickle
import copy
import random
import time
from collections import defaultdict

import numpy as np
import torch

from detzero_utils.common_utils import multi_processing

from detzero_refine.utils.data_utils import rotate_yaw


class DatasetTemplate(torch.utils.data.Dataset):
    def __init__(self, dataset_cfg=None, class_names=None, training=True, root_path=None, logger=None):
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.class_names = class_names
        self.training = training
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.logger = logger

        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = os.path.join(self.root_path, 'ImageSets', self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.augment_single = False if not self.training else dataset_cfg["AUGMENTOR"].get("AUGMENT_SINGLE", False)
        self.augment_full = False if not self.training else dataset_cfg["AUGMENTOR"].get("AUGMENT_FULL", False)
        self.tta = False if self.training else dataset_cfg.get("TTA", False)

        self.encoding = self.dataset_cfg.get('ENCODING', ['placeholder'])
        self.iou = None if not self.training else dataset_cfg.get("IOU_NAME", None)

        self.class_map = {
            'Vehicle': 1, 'Pedestrian': 2, 'Cyclist': 3,
            1: 'Vehicle', 2: 'Pedestrian', 3: 'Cyclist'
        }
        self.box_num = 0
        self.workers_num = 8

        self.total_epochs = 0
        self._merge_all_iters_to_one_epoch = False
        self.track_infos = []

    def init_infos(self, mode):
        """
        Function:
            Load sequence infos and load points
        """
        print('Loading Waymo Data for %s.' % self.split)
        self.data_infos = {}
        self.load_track_infos()
        self.sort_track_infos(self.data_infos)
        del self.data_infos
        
        track_num = len(self.veh_infos)+len(self.ped_infos)+len(self.cyc_infos)\
            if self.training and len(self.class_names) == 3 else len(self.track_infos)

        self.logger.info('Total Object Tracks for Waymo dataset: %d' % track_num)
        self.logger.info('Total Boxes for Waymo dataset: %d' % self.box_num)

    def load_track_infos(self):
        """
        Function:
            Load sequence infos and points from the pickle
            files with multi-processing logics
        """
        seq_ids = []
        for cls_name in self.class_names:
            data_path = os.path.join(self.root_path, 'refining', cls_name)
            file_names = os.listdir(data_path)

            for seq in self.sample_sequence_list:
                seq = seq.strip('segment-').strip('_with_camera_labels.tfrecord') + '.pkl'
                if seq in file_names:
                    seq_ids.append(os.path.join(data_path, seq))

        self.logger.info('Total Sequences: %d' % len(seq_ids))
        start_time = time.time()

        if self.iou:
            iou_path = os.path.join(self.root_path, 'refining', self.iou)
            with open(iou_path, 'rb') as f:
                self.iou = pickle.load(f)

        res = multi_processing(self.load_infos_worker, seq_ids, self.workers_num)
        for item in res:
            self.data_infos.update(item)

        end_time = time.time()
        log_str = 'Total cost time: {:.2f}min, Per seq cost time: {:.2f}s'.format(
            (end_time-start_time)/60, (end_time-start_time)/len(seq_ids))
        self.logger.info(log_str)

    def load_infos_worker(self, seq_name):
        data_infos = {}

        seq_infos = pickle.load(open(seq_name, 'rb'))
        obj_ids = list(seq_infos.keys())
        
        for obj_id in obj_ids:
            obj_info = seq_infos[obj_id]
            seq = obj_info['sequence_name']
            dict_key = seq + '/' + str(obj_id)

            # only use false positive object tracks for crm training
            mth_tk = obj_info.get('matched_tracklet', True)
            if self.training:
                if not mth_tk and not self.iou: continue
            else:
                if not mth_tk and not self.dataset_cfg.save_to_file: continue
            
            assert len(obj_info['boxes_global']) == len(obj_info['gt_boxes_global'])

            data_infos[dict_key] = obj_info

            if self.iou is not None:
                data_infos[dict_key]['refine_iou'] = self.iou[seq][obj_id]
            else:
                data_infos[dict_key]['refine_iou'] = np.zeros(len(obj_info['sample_idx']))

        return data_infos

    def sort_track_infos(self, data_infos):
        if self.training and len(self.class_names) == 3:
            self.veh_infos, self.ped_infos, self.cyc_infos = [], [], []
            
            for key, val in data_infos.items():
                self.box_num += len(val['boxes_global'])
                
                if val['name'] == 'Vehicle':
                    self.veh_infos.append(val)
                elif val['name'] == 'Pedestrian':
                    self.ped_infos.append(val)
                elif val['name'] == 'Cyclist':
                    self.cyc_infos.append(val)

        else:
            for key, val in data_infos.items():
                self.box_num += len(val['boxes_global'])
                self.track_infos.append(val)

    @property
    def mode(self):
        return 'train' if self.training else 'test'

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.track_infos) * self.total_epochs
        if self.training and len(self.class_names) == 3:
            return len(self.veh_infos) + len(self.ped_infos) + len(self.cyc_infos)*50
        else:
            return len(self.track_infos)

    def merge_all_iters_to_one_epoch(self, merge=True, epochs=None):
        if merge:
            self._merge_all_iters_to_one_epoch = True
            self.total_epochs = epochs
        else:
            self._merge_all_iters_to_one_epoch = False

    def prepare_data(self, data_dict):

        if self.tta:
            data_dict = self.tta_operator(data_dict)

        return data_dict

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.track_infos)

        if self.training and len(self.class_names) == 3:
            if index < len(self.veh_infos):
                data_info = self.veh_infos[index]
            
            elif index >= len(self.veh_infos) and index < len(self.veh_infos) + len(self.ped_infos):
                new_index = np.random.randint(len(self.ped_infos))
                data_info = self.ped_infos[new_index]
            
            else:
                new_index = np.random.randint(len(self.cyc_infos))
                data_info = self.cyc_infos[new_index]
        
        else:
            data_info = copy.deepcopy(self.track_infos[index])

        track_feats = self.extract_track_feature(data_info)

        data_dict = {}
        data_dict.update(track_feats)

        data_dict = self.prepare_data(data_dict=data_dict)

        return data_dict

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        batch_size = len(batch_list)
        tta = 'tta_original' in batch_list[0]

        for cur_sample in batch_list:
            if tta:
                tta_ops = cur_sample.keys()
                data_dict['tta_ops'] = [tta_cfg for tta_cfg in tta_ops]
                for key in cur_sample["tta_original"]:
                    if key in ['geo_query_points', 'geo_memory_points', 'geo_query_boxes',
                               'geo_query_num', 'pos_query_points', 'pos_memory_points',
                               'pos_trajectory', 'padding_mask', 'conf_points']:
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
                if key in ['obj_cls', 'geo_memory_points',
                           'pos_init_box', 'pos_query_points', 'pos_memory_points',
                           'pos_trajectory', 'gt_pos_trajectory', 'padding_mask',
                           'iou', 'conf_score', 'conf_points']:
                    ret[key] = np.stack(val, axis=0)

                elif key in ['geo_query_points']:
                    max_len = max(data_dict['geo_query_num'])
                    temp = []
                    for i, pts in enumerate(val):
                        pts = np.array(pts)
                        pts_pad = np.zeros([max_len-pts.shape[0], pts.shape[1], pts.shape[2]])
                        pts_pad = np.concatenate([pts, pts_pad], axis=0)
                        temp.append(pts_pad)
                    ret[key] = np.stack(temp, axis=0)

                elif key in ['geo_query_boxes', 'gt_geo_query_boxes']:
                    max_len = max(data_dict['geo_query_num'])
                    temp = []
                    for i, box in enumerate(val):
                        box_pad = np.zeros([max_len-box.shape[0], box.shape[1]])
                        box_pad = np.concatenate([box, box_pad], axis=0)
                        temp.append(box_pad)
                    ret[key] = np.stack(temp, axis=0)
                
                else:
                    ret[key] = val
            except:
                print('Error in collate_batch: key=%s' % key)
                raise TypeError

        ret['batch_size'] = batch_size if not tta else int(batch_size * len(tta_ops))
        return ret


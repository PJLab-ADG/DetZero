from collections import defaultdict
import os
import pickle

import numpy as np
 
from detzero_utils.common_utils import get_log_info

from detzero_track.utils.data_utils import sequence_list_to_dict
from .dataset import DatasetTemplate


class WaymoTrackDataset(DatasetTemplate):
    """
    The Dataset class for Tracking on Waymo
    """
    def __init__(self, dataset_cfg, data_path, log_time, split,
                 root_path=None, logger=None):
        super().__init__(dataset_cfg, data_path, split, root_path, logger)
        if self.logger:
            self.logger.info(get_log_info('Initialize {}'.format(self.dataset_cfg.DATASET)))

        self.logtime = log_time

        self.load_paths()
        self.init_infos()

    def load_paths(self):
        """
        Load all submodule save paths. 
        """
        self.track_module_path = defaultdict(dict)

        # set the desitation path of tracking result
        track_path = os.path.join(self.root_path, 'tracking')
        if not os.path.exists(track_path):
            os.makedirs(track_path)
        
        self.track_module_path['tracking'] = os.path.join(
            track_path, '-'.join(['tracking', self.split, self.logtime]))
        self.track_module_path['det_drop'] = os.path.join(
            track_path, '-'.join(['drop', self.split, self.logtime])
        )

        # set the path of detection prediction results
        self.track_module_path['detection'] = self.det_path

        # set the path of ground-truth infos
        self.gt_path = os.path.join(self.root_path, 'waymo_infos_%s.pkl' % self.split)

    def init_infos(self):
        with open(self.track_module_path['detection'], 'rb') as f:
            raw_det = pickle.load(f)

        if isinstance(raw_det, list):
            det_info = sequence_list_to_dict(raw_det)
        elif isinstance(raw_det, dict):
            det_info = raw_det
        else:
            raise TypeError('The format of detection results must be List or Dict.')

        self.seq_name_list = list(det_info.keys())[:]
        self.seq_det_infos = [det_info[seq_n] for seq_n in self.seq_name_list]

        if self.assign_mode:
            with open(self.gt_path, 'rb') as f:
                raw_gt_infos = pickle.load(f)

            gt_infos = sequence_list_to_dict(raw_gt_infos)
            if len(gt_infos.keys()) != len(self.seq_name_list):
                raise ValueError('Ground-truth infomation loading falied.')
            self.gt_infos = [gt_infos[seq_n] for seq_n in self.seq_name_list]

    def __len__(self):
        return len(self.seq_name_list)

    def __getitem__(self, idx):
        seq_name = self.seq_name_list[idx]
        det_infos = self.seq_det_infos[idx]

        det_data, drop_data = self.data_processor.forward(data_dict=det_infos)
        data_dict = {'detection': det_data, 'det_drop': drop_data}
        if self.assign_mode:
            data_dict.update({'gt': self.gt_infos[idx]})
        return seq_name, data_dict

    @staticmethod
    def collate_batch(batch_list):
        seq_names = []
        seq_data_dicts = defaultdict(list)
        
        for i, cur_sample in enumerate(batch_list):
            seq_names.append(cur_sample[0])
            for k, v in cur_sample[1].items():
                seq_data_dicts[k].append(v)

        return seq_names, seq_data_dicts

    def get_track_path(self):
        """
        Get tracking path. 
        """
        return self.track_module_path['tracking'] + '.pkl'
    
    def get_drop_path(self):
        """
        Get tracking path. 
        """
        return self.track_module_path['det_drop'] + '.pkl'

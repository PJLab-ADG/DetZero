import os
import time
import pickle
import copy
import random
from pathlib import Path

import numpy as np

from detzero_utils.box_utils import boxes_to_corners_3d

from detzero_refine.datasets.dataset import DatasetTemplate
from detzero_refine.utils.data_utils import sample_points, init_coords_transform, world_to_lidar


class WaymoConfidenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)

        self.init_infos(self.mode)
        self.query_num = self.dataset_cfg.get('QUERY_NUM', 200)
        self.query_pts_num = self.dataset_cfg.get('QUERY_POINTS_NUM', 256)
        
        self.pos_tk_infos, self.neg_tk_infos = [], []
        for tk in self.track_infos:
            if tk['matched_tracklet'] == True:
                self.pos_tk_infos.append(tk)
            else:
                self.neg_tk_infos.append(tk)

        self.logger.info('Positive tracks: {0}, Negative tracks num: {1}'.format(
            len(self.pos_tk_infos), len(self.neg_tk_infos)))

    def __len__(self):
        if self.training:
            return len(self.pos_tk_infos) * 2
        else:
            return len(self.track_infos)

    def __getitem__(self, index):
        if self.training:
            if index % 2 == 0:
                data_info = copy.deepcopy(self.pos_tk_infos[index//2])
            else:
                new_index = np.random.randint(len(self.neg_tk_infos))
                data_info = copy.deepcopy(self.neg_tk_infos[new_index])
        else:
            data_info = copy.deepcopy(self.track_infos[index])
        
        track_feats = self.extract_track_feature(data_info)

        input_dict = {}
        input_dict.update(track_feats)

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict

    def extract_track_feature(self, data_info):
        traj_all = data_info['boxes_global']
        score_all = data_info['score']
        frame_id_all = data_info['sample_idx']
        pts_all = data_info['pts']
        iou_all = data_info.get('refine_iou', None)
        
        # randomly sample the object track
        if self.training:
            traj_len = len(traj_all)
            samples = random.sample(
                range(traj_len),
                random.randint(min(5, traj_len), traj_len)
            )
            
            score = score_all[samples]
            frame_id = frame_id_all[samples]
            traj = traj_all[samples]
            iou = iou_all[samples]
            pts = [pts_all[ind] for ind in samples]

        else:
            pts = pts_all
            traj = traj_all
            frame_id = frame_id_all
            score = score_all
            iou = iou_all


        if self.training:
            sample_idx = np.random.randint(0, len(traj))
        else:
            sample_idx = (len(traj)) // 2

        init_box = copy.deepcopy(traj[sample_idx, 0:7])
        # transform trajectory to init box coordinate
        init_box, pts, traj, _ = init_coords_transform(
            init_box, pts, traj)

        box_num = len(traj)

        sample_pts = []
        for i in range(len(pts)):
            pts_i = pts[i]
            sa_pts_i = sample_points(
                pts_i, sample_num=self.query_pts_num, replace=False)
            sample_pts.append(sa_pts_i)
        pts = np.stack(sample_pts, axis=0)

        input_pts_data = []
        for enc_cfg in self.encoding:
            if enc_cfg == 'placeholder':
                input_pts_data.append(pts)
                break

            elif enc_cfg == 'xyz':
                input_pts_data.append(pts[:, :, :3])

            elif enc_cfg == 'intensity':
                input_pts_data.append(pts[:, :, [3]])

            elif enc_cfg == 'p2co':
                co_pts = boxes_to_corners_3d(traj).reshape(box_num, -1)
                co_ce_pts = np.concatenate([co_pts, traj[:, :3]], axis=-1)
                p2co_feat = np.tile(pts[:, :, :3], (1, 1, 9)) -\
                    np.tile(co_ce_pts[:, None, :], (1, self.query_pts_num, 1))
                input_pts_data.append(p2co_feat)

            elif enc_cfg == 'box_pos':
                box_pos = np.concatenate([traj[:, :3], traj[:, 6:7]], axis=-1)[:, None, :]
                box_pos = np.tile(box_pos, (1, self.query_pts_num, 1))
                input_pts_data.append(box_pos)

            elif enc_cfg == 'score':
                score_feat = np.tile(score[:, None, None], (1, self.query_pts_num, 1))
                input_pts_data.append(score_feat)

            else:
                raise NotImplementedError

        input_pts_data = np.concatenate(input_pts_data, axis=2)
        input_pts_data = np.concatenate([
            input_pts_data,
            np.zeros((self.query_num-box_num, self.query_pts_num, input_pts_data.shape[2]))],
            axis=0
        )

        # pad to a same track length
        iou = np.concatenate((iou, np.full(self.query_num-len(iou), -1)), axis=0)
        score = np.concatenate((score, np.full(self.query_num-len(score), -1)), axis=0)

        obj_info = {
            'sequence_name': data_info['sequence_name'],
            'frame': frame_id,
            'obj_id': data_info['obj_id'],
            'conf_score': score,
            'state': data_info['state'],
            'matched_tracklet': data_info['matched_tracklet'],
            'iou': iou,
            'box_num': box_num,
            'conf_points': input_pts_data
        }

        return obj_info

    def generate_prediction_dicts(self, batch_dict, pred_dicts, single_pred_dict, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_score: (N), Tensor
            single_pred_dict: dict, to save the result back

        Returns:
        """
        annos = []

        for i in range(batch_dict['batch_size']):
            seq = batch_dict['sequence_name'][i]
            obj_id = batch_dict['obj_id'][i]
            box_num = int(batch_dict['box_num'][i])

            frm_id = batch_dict['frame'][i][:box_num].astype(np.int)
            score = batch_dict['conf_score'].cpu().numpy()[i][:box_num]
            pred_score = pred_dicts['pred_score'][i][:box_num]

            if seq not in single_pred_dict:
                single_pred_dict[seq] = {}

            single_pred_dict[seq][obj_id] = {
                'sequence_name': seq,
                'frame_id': frm_id,
                'score': score,
                'new_score': pred_score
            }

        return annos

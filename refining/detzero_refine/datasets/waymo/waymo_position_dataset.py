import os
import pickle
import copy
import random
from pathlib import Path

import numpy as np

from detzero_utils import common_utils
from detzero_utils.box_utils import boxes_to_corners_3d

from detzero_refine.datasets.dataset import DatasetTemplate
from detzero_refine.utils.position_augment import augment_full_track, test_time_augment
from detzero_refine.utils.data_utils import (sample_points,
                                             world_to_lidar,
                                             init_coords_transform,
                                             box_coords_transform)


class WaymoPositionDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(dataset_cfg, class_names, training, root_path, logger)
        
        self.init_infos(self.mode)

        self.query_num = self.dataset_cfg.get('QUERY_NUM', 200)
        self.query_pts_num = self.dataset_cfg.get('QUERY_POINTS_NUM', 256)
        self.memory_pts_num = self.dataset_cfg.get('MEMORY_POINTS_NUM', 48)


    def extract_track_feature(self, data_info):
        obj_cls = self.class_map[data_info['name']]
        traj_all = data_info['boxes_global']
        score_all = data_info['score']
        frame_id_all = data_info['sample_idx']
        pose_all = data_info['pose']
        pts_all = data_info['pts']
        matched = data_info['matched']
        mth_tk = data_info['matched_tracklet']
        state = data_info['state']
        if data_info.get('gt_boxes_global', None) is not None:
            traj_gt_all = data_info['gt_boxes_global'][:, :7]

        # randomly sample the object track
        if self.training:
            traj_len = len(traj_all[matched])
            # the result would be orderless
            samples = random.sample(
                range(traj_len),
                random.randint(min(5, traj_len), traj_len)
            )
            
            score = score_all[matched][samples]
            pose = pose_all[matched][samples]
            frm_id = frame_id_all[matched][samples]
            traj = traj_all[matched][samples]
            traj_gt = traj_gt_all[matched][samples]

            pts_mth = [pts_all[i] for i in range(len(traj_all)) if matched[i]]
            pts = [pts_mth[ind] for ind in samples]

        else:
            samples = np.arange(len(traj_all))
            score = score_all
            pose = pose_all
            frm_id = frame_id_all
            traj = traj_all
            traj_gt = traj_gt_all
            pts = pts_all

        # randomly select the inital coordinate origin
        if self.training:
            sample_idx = np.random.randint(0, len(traj))
        else:
            sample_idx = (len(traj)) // 2
        init_box = traj[sample_idx, 0:7].copy()
        
        # transform traj and traj_gt to init box coordinate
        init_box, pts, traj, traj_gt = init_coords_transform(
            init_box, pts, traj, traj_gt)

        # gt_box = traj_gt[sample_idx, 0:7].copy()
        box_num = len(traj)

        # for each proposal, randomly sample 256 points as query,
        #   randomly sample 48 points as dense trajectory
        query_pts, traj_pts = [], []
        for i in range(len(pts)):
            pts_sa = sample_points(pts[i], sample_num=self.query_pts_num)
            query_pts.append(pts_sa)

            pts_sa = sample_points(pts[i], sample_num=self.memory_pts_num)
            traj_pts.append(pts_sa)

        query_pts = np.stack(query_pts, axis=0)
        traj_pts = np.stack(traj_pts, axis=0)

        local_pts_data = []
        global_pts_data = []
        for enc_cfg in self.encoding:
            if enc_cfg == 'placeholder':
                local_pts_data.append(query_pts)
                global_pts_data.append(traj_pts)
                break

            elif enc_cfg == 'xyz':
                local_pts_data.append(query_pts[:, :, :3])
                global_pts_data.append(traj_pts[:, :, :3])

            elif enc_cfg == 'intensity':
                local_pts_data.append(query_pts[:, :, [3]])
                global_pts_data.append(traj_pts[:, :, [3]])

            elif enc_cfg == 'p2co':
                co_pts = boxes_to_corners_3d(traj).reshape(box_num, -1)
                co_ce_pts = np.concatenate([co_pts, traj[:, :3]], axis=-1)
                
                p2co_feat = np.tile(query_pts[:, :, :3], (1, 1, 9)) -\
                    np.tile(co_ce_pts[:, None, :], (1, self.query_pts_num, 1))
                local_pts_data.append(p2co_feat)

                p2co_feat = np.tile(traj_pts[:, :, :3], (1, 1, 9)) -\
                    np.tile(co_ce_pts[:, None, :], (1, self.memory_pts_num, 1))
                global_pts_data.append(p2co_feat)

            elif enc_cfg == 'score':
                local_pts_data.append(np.tile(score[:, None, None], (1, self.query_pts_num, 1)))
                global_pts_data.append(np.tile(score[:, None, None], (1, self.memory_pts_num, 1)))

            elif enc_cfg == 'class':
                tmp_cat = np.zeros(3)
                tmp_cat[obj_cls - 1] = 1
                local_pts_data.append(np.tile(tmp_cat[None, None, :], (box_num, self.query_pts_num, 1)))
                global_pts_data.append(np.tile(tmp_cat[None, None, :], (box_num, self.memory_pts_num, 1)))
                
            else:
                raise NotImplementedError

        local_pts_data = np.concatenate(local_pts_data, axis=2)
        global_pts_data = np.concatenate(global_pts_data, axis=2)

        # do augmentation after all the distance related calculation
        if self.training and self.augment_full:
            query_pts, traj_pts, traj, traj_gt = augment_full_track(
                query_pts, traj_pts, traj, traj_gt)
        
        # pad the box
        local_pts_data = np.concatenate([
            local_pts_data,
            np.zeros((self.query_num-box_num, self.query_pts_num, local_pts_data.shape[2]))],
            axis=0
        )
        global_pts_data = np.concatenate([
            global_pts_data,
            np.zeros((self.query_num-box_num, self.memory_pts_num, global_pts_data.shape[2]))],
            axis=0
        )

        zeros = np.zeros((self.query_num-len(traj_gt), 7), dtype=np.float32)
        traj_gt = np.concatenate((traj_gt[:, :7], copy.deepcopy(zeros)), axis=0)
        traj = np.concatenate((traj[:, :7], copy.deepcopy(zeros)), axis=0)
        padding_mask = np.concatenate(
            (np.zeros(box_num), np.full(self.query_num-box_num, 1)), axis=0)

        obj_info = {
            'sequence_name': data_info['sequence_name'],
            'frame': frm_id,
            'obj_id': data_info['obj_id'],
            'obj_cls': obj_cls,
            'pos_trajectory': traj,
            'gt_pos_trajectory': traj_gt,
            'pos_scores': score,
            'pos_init_box': init_box,
            'box_num': box_num,
            'padding_mask': padding_mask,
            'pos_query_points': local_pts_data,
            'pos_memory_points': global_pts_data,
            'pose': pose,
            'state': state,
            'matched': matched,
            'matched_tracklet': mth_tk
        }

        return obj_info

    @staticmethod
    def tta_operator(data_dict):
        return test_time_augment(data_dict)

    def generate_prediction_dicts(self, batch_dict, pred_dicts, single_pred_dict, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            single_pred_dict: dict, to save the result back

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_boxes = box_dict['pred_boxes']
            pred_dict = get_template_prediction(pred_boxes.shape[0])
            if pred_boxes.shape[0] == 0:
                print('-------------------ERROR--------------------')
                return pred_dict

            return self.revert_to_each_frame(box_dict)

        annos = []

        all_pred_res, all_gt_res, all_pred_world_res, all_gt_world_res = generate_single_sample_dict(pred_dicts)

        for per_obj in range(len(all_pred_res)):
            seq = batch_dict['sequence_name'][per_obj]
            obj_id = batch_dict['obj_id'][per_obj]
            
            if seq not in single_pred_dict:
                single_pred_dict[seq] = {}
            
            single_pred_dict[seq][obj_id] = {
                'sequence_name': seq,
                'frame_id': [],
                'boxes_lidar': [],
                'boxes_global': [],
                'score': [],
                'name': [],
                'state': batch_dict['state'][per_obj],
                'pose': [],
                'boxes_gt': [],
                'boxes_gt_global': []
            }

            # check whether to output the gt related result
            for idx, frm_id in enumerate(batch_dict['frame'][per_obj]):
                single_pred_dict[seq][obj_id]['boxes_lidar'].append(all_pred_res[per_obj][idx])
                single_pred_dict[seq][obj_id]['score'].append(batch_dict['pos_scores'][per_obj][idx])
                single_pred_dict[seq][obj_id]['name'].append(self.class_map[int(batch_dict['obj_cls'][per_obj])])
                single_pred_dict[seq][obj_id]['pose'].append(pred_dicts['pose'][per_obj][idx])
                single_pred_dict[seq][obj_id]['frame_id'].append(int(frm_id))
                single_pred_dict[seq][obj_id]['boxes_gt'].append(all_gt_res[per_obj][idx])
                single_pred_dict[seq][obj_id]['boxes_global'].append(all_pred_world_res[per_obj][idx])
                single_pred_dict[seq][obj_id]['boxes_gt_global'].append(all_gt_world_res[per_obj][idx])

        return annos

    def revert_to_each_frame(self, data_dict):
        seq_lidar = []
        seq_world = []
        seq_lidar_gt = []
        seq_world_gt = []

        for i in range(len(data_dict['pred_boxes'])):
            box_preds_world = data_dict['pred_boxes'][i].copy()
            init_box = data_dict['pos_init_box'][i]
            pose = data_dict['pose'][i]
            frm_len = len(pose)

            box_gt_world = data_dict['gt_pos_trajectory'][i].copy()

            # convert the predicted boxes from init_box coords to global coords
            box_preds_world = box_coords_transform(box_preds_world, init_box)
            seq_world.append(box_preds_world[:frm_len, :].copy())

            # convert the predicted boxes from global coords to lidar frame coords
            box_preds_lidar = world_to_lidar(box_preds_world[:frm_len, :], pose)
            seq_lidar.append(box_preds_lidar)

            # the same process for gt_box
            box_gt_world = box_coords_transform(box_gt_world, init_box)
            seq_world_gt.append(box_gt_world[:frm_len, :].copy())
            
            box_gt_lidar = world_to_lidar(box_gt_world[:frm_len, :], pose)
            seq_lidar_gt.append(box_gt_lidar)

        return seq_lidar, seq_lidar_gt, seq_world, seq_world_gt

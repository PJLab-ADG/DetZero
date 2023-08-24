import os
import pickle

from functools import partial
from collections import defaultdict
import numpy as np

from detzero_utils.common_utils import create_logger, get_log_info, multi_processing

from detzero_track.utils.data_utils import (sequence_list_to_dict,
                                            frame_list_to_dict,
                                            tracklets_to_frames)
from detzero_track.utils.track_calculation import (get_trajectory_similarity,
                                                   get_iou_mat_dict,
                                                   get_gt_id_data)
from detzero_track.models.tracking_modules.data_association import GNN_assignment


class TrackRecall():
    def __init__(self, root_path, data_path, split, workers, class_names, 
                 difficultys=['l2'], iou_threshold=[0.7, 0.5, 0.5], method='3d', 
                 logger=None):
        self.root_path = root_path
        self.data_path = data_path
        self.split = split
        self.workers = workers
        self.class_names = class_names
        self.difficultys = difficultys
        self.method = method

        assert len(class_names) == len(iou_threshold)
        self.iou_thresholds = dict()
        for idx, class_n in enumerate(self.class_names):
            self.iou_thresholds[class_n] = iou_threshold[idx]

        if logger is None:
            self.logger = create_logger()
        else:
            self.logger = logger

        self.logger.info(get_log_info('Begin Tracklet Recall Evaluation'))
        self.match_rate_list = np.arange(0, 10) * 0.1

        self.init_data()

    def init_data(self):
        self.gt_path = os.path.join(
            self.root_path, 'data/waymo', 'waymo_infos_%s.pkl' % self.split)
        
        with open(self.gt_path, 'rb') as f:
            raw_gt_data = pickle.load(f)
        gt_data = sequence_list_to_dict(raw_gt_data)

        with open(self.data_path, 'rb') as f:
            pred_data = pickle.load(f)

        self.seq_name_list = list(pred_data.keys())
        self.gt_data = gt_data
        self.pred_data = pred_data

    def get_tracklet_recall(self):
        eval_single_seq = partial(self.eval_single_seq, class_names=self.class_names,
                                  difficultys=self.difficultys, iou_thresholds=self.iou_thresholds,
                                  method=self.method)
        input_data_dict = [
            {'gt': self.gt_data[x], 'pred': self.pred_data[x]} for x in self.seq_name_list
        ]
        eval_results = multi_processing(
            eval_single_seq, input_data_dict, workers=self.workers, bar=True
        )
        eval_results = dict(zip(self.seq_name_list, eval_results))

        tracklet_recall_result = self.calculate_tracklet_recall(eval_results)
        tracklet_recall_result['iou_thresholds'] = self.iou_thresholds
        tracklet_recall_result['method'] = self.method

        self.logger.info(get_log_info('Tracklet Recall Results'))
        self.show_traj_ap_infos(tracklet_recall_result)

    def show_traj_ap_infos(self, trk_recall_result):
        show_attributes = ['cutoffs', 'recalls', 'tp', 'fp', 'pred_nums', 'gt_nums']
        for class_n in self.class_names:
            for difficulty in self.difficultys:
                scores_len = len(trk_recall_result[difficulty][class_n]['tp'])
                for i in range(scores_len):
                    if i != 8: continue
                    log_str = '{:<3s} {:<10s}'.format(difficulty.upper(), class_n)
                    for attribute in show_attributes:
                        val = trk_recall_result[difficulty][class_n][attribute][i]
                        if attribute in ['tp', 'fp']:
                            log_str += '{}{} {:<4.1f}'.format(' '*2, 
                                attribute[:].upper()+':', val/len(self.seq_name_list))
                        elif attribute in ['pred_nums', 'gt_nums']:
                            log_str += '{}{} {:<4.1f}'.format(' '*2, 
                                attribute[:].capitalize()+':', val/len(self.seq_name_list))
                        else:
                            log_str += '{}{} {:<.4f}'.format(' '*2, 
                                attribute[:-1].capitalize()+':', val)

                    self.logger.info(log_str)

    def calculate_tracklet_recall(self, eval_result):
        temp_eval_result = defaultdict(dict)
        for difficulty in self.difficultys:
            for class_n in self.class_names:
                temp_eval_result[difficulty][class_n] = defaultdict(list)
                for seq_n, result in eval_result.items():    
                    for key, val in eval_result[seq_n][difficulty][class_n].items():
                        temp_eval_result[difficulty][class_n][key].extend(val)

        ap_result = defaultdict(dict)
        for difficulty in self.difficultys:
            for class_n in self.class_names:
                ap_result[difficulty][class_n] = defaultdict(list)

                for key in temp_eval_result[difficulty][class_n].keys():
                    if key == 'duplicate_preds':
                        temp_eval_result[difficulty][class_n][key] = \
                            np.array(temp_eval_result[difficulty][class_n][key], dtype=object)
                    else:
                        temp_eval_result[difficulty][class_n][key] = \
                            np.array(temp_eval_result[difficulty][class_n][key])

        for difficulty in self.difficultys:
            for class_n in self.class_names:
                gt_nums = len(temp_eval_result[difficulty][class_n]['gt_box_nums_list'])
                pred_nums = len(temp_eval_result[difficulty][class_n]['match_pred_box_nums_list']) + \
                    len(temp_eval_result[difficulty][class_n]['unmatch_pred_box_nums_list'])

                for match_rate_threshold in self.match_rate_list:
                    if gt_nums > 0:                    
                        score_mask = temp_eval_result[difficulty][class_n]['match_rate'] >= match_rate_threshold
                        tp = np.count_nonzero(score_mask)
                    else:
                        tp = 0

                    fn = gt_nums - tp
                    fp = pred_nums - tp
                    precision = tp/(tp+fp+1e-9)
                    recall = tp/(gt_nums+1e-9)

                    ap_result[difficulty][class_n]['precisions'].append(precision)
                    ap_result[difficulty][class_n]['recalls'].append(recall)
                    ap_result[difficulty][class_n]['tp'].append(tp)
                    ap_result[difficulty][class_n]['fp'].append(fp)
                    ap_result[difficulty][class_n]['pred_nums'].append(pred_nums)
                    ap_result[difficulty][class_n]['cutoffs'].append(float(match_rate_threshold))
                    ap_result[difficulty][class_n]['gt_nums'].append(gt_nums)

        return ap_result


    @staticmethod
    def eval_single_seq(data_dict, class_names, difficultys, iou_thresholds, method='3d'):
        """
        Evaluate the tracking performance of a single sequence.

        Args:
            data_dict (dict): A dictionary containing the ground truth and predicted tracklets.
            class_names (list): A list of class names.
            difficultys (list): A list of difficulty levels.
            iou_thresholds (list): A list of IoU thresholds.
            method (str): The method used to compute IoU. Default is '3d'.

        Returns:
            dict: A dictionary containing the evaluation results for each difficulty level, class name and object.
        """
        gt_data = data_dict['gt']
        pred_data = data_dict['pred']

        frame_pred_data = tracklets_to_frames({'reference':gt_data, 'source':pred_data})
        # assert len(gt_data) == len(frame_pred_data), 'frame length of gt != pred data'

        dict_frame_pred_data = frame_list_to_dict(frame_pred_data)
        iou_mat_dict = get_iou_mat_dict(
            gt_data, dict_frame_pred_data, class_names, distinguish_class=True, iou=method)

        gt_keys = ['gt_boxes_global', 'name', 'obj_ids', 'difficulty', 'num_points_in_gt']
        gt_id_data = get_gt_id_data(gt_data, gt_keys, class_names)

        for item in frame_pred_data:
            for iou_idx, obj_id in enumerate(item['obj_ids']):
                pred_data[obj_id]['iou_idx'].append(iou_idx)

        gt_ids = list(gt_id_data.keys())
        pred_ids = list(pred_data.keys())
        traj_similar_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
        traj_same_count_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
        traj_match_count_mat = np.zeros((len(gt_ids), len(pred_ids)), dtype=np.float32)
        for gt_id_idx, gt_id in enumerate(gt_ids):
            gt_info = gt_id_data[gt_id]
            for key in gt_info.keys():
                gt_info[key] = np.array(gt_info[key])
            for pred_id_idx, pred_id in enumerate(pred_ids):
                pred_info = pred_data[pred_id]
                pred_info['iou_idx'] = np.array(pred_info['iou_idx'])

                similarity, match_count, same_frame_count = get_trajectory_similarity(
                    gt_info, pred_info, iou_mat_dict, iou_thresholds, 0.
                )
                traj_similar_mat[gt_id_idx, pred_id_idx] = similarity
                traj_same_count_mat[gt_id_idx, pred_id_idx] = same_frame_count
                traj_match_count_mat[gt_id_idx, pred_id_idx] = match_count

        match, unmatch_gt, unmatch_pred = GNN_assignment(1-traj_similar_mat)

        total_gt_traj_nums = len(gt_ids)
        total_match_traj_nums = len(match)

        eval_result = defaultdict(dict)
        for difficulty in difficultys:
            for class_n in class_names:
                eval_result[difficulty][class_n] = defaultdict(list)

        for gt_id, val in gt_id_data.items():
            difficulty = np.array(gt_id_data[gt_id]['difficulty'])
            num_points_in_gt = np.array(gt_id_data[gt_id]['num_points_in_gt'])
            l1_difficulty_mask = (num_points_in_gt > 5) & (difficulty == 0)
            l2_difficulty_mask = (num_points_in_gt <= 5) & (difficulty == 0)
            difficulty[l1_difficulty_mask] = 1
            difficulty[l2_difficulty_mask] = 2
            gt_id_data[gt_id]['difficulty'] = difficulty

        difficulty_dict = {1:'l1', 2:'l2'}
        for match_idx in range(len(match)):
            gt_idx = match[match_idx, 0]
            pred_idx = match[match_idx, 1]

            gt_id = gt_ids[gt_idx]
            name = gt_id_data[gt_id]['name'][0]
            difficulty = gt_id_data[gt_id]['difficulty']
            if np.count_nonzero(difficulty == 2) > 0:
                obj_difficulty = 'l2'
            else: obj_difficulty = 'l1'

            # obj_difficulty = difficulty_dict[difficulty[0]]
            eval_result[obj_difficulty][name]['gt_box_nums_list'].append(len(difficulty))
            eval_result[obj_difficulty][name]['match_rate'].append(traj_match_count_mat[gt_idx, pred_idx]/len(difficulty))

            pred_id = pred_ids[pred_idx]
            eval_result[obj_difficulty][name]['match_pred_box_nums_list'].append(len(pred_data[pred_id]['name']))

            eval_result[obj_difficulty][name]['match_len'].append(traj_match_count_mat[gt_idx, pred_idx])
            eval_result[obj_difficulty][name]['same_len'].append(traj_same_count_mat[gt_idx, pred_idx])
            eval_result[obj_difficulty][name]['duplicate_preds'].append(traj_match_count_mat[gt_idx, :])

        for unmatch_pred_idx in unmatch_pred:
            pred_id = pred_ids[unmatch_pred_idx]
            name = pred_data[pred_id]['name'][0]

            for difficulty in difficultys:
                eval_result[difficulty][name]['unmatch_pred_box_nums_list'].append(len(pred_data[pred_id]['name']))

        for unmatch_gt_idx in unmatch_gt:
            gt_id = gt_ids[unmatch_gt_idx]
            name = gt_id_data[gt_id]['name'][0]
            difficulty = gt_id_data[gt_id]['difficulty']
            obj_difficulty = difficulty_dict[difficulty[0]]

            eval_result[obj_difficulty][name]['gt_box_nums_list'].append(len(difficulty))

        return eval_result

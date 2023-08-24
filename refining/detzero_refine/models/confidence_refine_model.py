import copy

import numpy as np
import torch
import torch.nn as nn

from detzero_utils.ops.iou3d_nms import iou3d_nms_utils

from .refine_template import RefineTemplate


class ConfidenceRefineModel(RefineTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg, dataset)

    def post_processing(self, data_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = data_dict['batch_size']
        recall_dict = {}

        if self.model_cfg.POST_PROCESSING.get('GENERATE_RECALL', True):
            recall_dict = self.generate_recall_record(
                box_preds=None,
                recall_dict=recall_dict,
                data_dict=data_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        pred_dicts = {
            'pred_score': data_dict['pred_score'].cpu().numpy()
        }

        return pred_dicts, recall_dict

    @staticmethod
    def generate_recall_record(
            box_preds,
            recall_dict,
            data_dict=None,
            thresh_list=None):

        if 'iou' not in data_dict:
            return recall_dict
        
        bs = data_dict['batch_size']
        box_num = data_dict['box_num']
        state = data_dict['state']
        mth_tk = np.array(data_dict['matched_tracklet'])

        if recall_dict.__len__() == 0:
            recall_dict = {
                'Box num': 0,
                'Track num': 0,
                'static': 0,
                'dynamic': 0,
                'matched_up': 0,
                'matched_down': 0,
                'unmatched_up': 0,
                'unmatched_down': 0,
                'gt_score': [],
                'pred_score': [],
                'ori_score': []
            }
            
            for cur_th in thresh_list:
                recall_dict.update({
                    'Box level input %s' % str(cur_th): 0,
                    'Box level output %s' % str(cur_th): 0,
                    'Box level input (static) %s' % str(cur_th): 0,
                    'Box level output (static) %s' % str(cur_th): 0,
                    'Box level input (dynamic) %s' % str(cur_th): 0,
                    'Box level output (dynamic) %s' % str(cur_th): 0,
                    'Track level input %s' % str(cur_th): 0,
                    'Track level output %s' % str(cur_th): 0,
                    'Track level input (static) %s' % str(cur_th): 0,
                    'Track level output (static) %s' % str(cur_th): 0,
                    'Track level input (dynamic) %s' % str(cur_th): 0,
                    'Track level output (dynamic) %s' % str(cur_th): 0
                })
        
        for i in range(bs):
            num = box_num[i]
            ori_score = data_dict['conf_score'][i][:num].cpu().numpy()
            pred_score = data_dict['pred_score'][i][:num].cpu().numpy()
            gt_score = data_dict['iou'][i][:num].cpu().numpy()
            
            gt_label = gt_score >= 0.7

            recall_dict['ori_score'].extend(ori_score)
            recall_dict['pred_score'].extend(pred_score)
            recall_dict['gt_score'].extend(gt_label)
            
            matched_up = gt_label & (pred_score >= ori_score)
            recall_dict['matched_up'] += matched_up.sum()
            
            matched_down = gt_label & (pred_score < ori_score)
            recall_dict['matched_down'] += matched_down.sum()

            unmatched_up = (gt_score < 0.7) & (pred_score >= ori_score)
            recall_dict['unmatched_up'] += unmatched_up.sum()

            recall_dict['unmatched_down'] += num - matched_up.sum() - matched_down.sum() - unmatched_up.sum()

            recall_dict['Box num'] += num
        
        recall_dict['Track num'] += np.where(mth_tk == True)[0].shape[0]

        return recall_dict


import numpy as np
import torch
import torch.nn as nn

from detzero_utils.ops.iou3d_nms import iou3d_nms_utils

from .refine_template import RefineTemplate


class GeometryRefineModel(RefineTemplate):
    def __init__(self, model_cfg, dataset):
        super().__init__(model_cfg, dataset)

    def post_processing(self, data_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        recall_dict = {}

        batch_box_preds = data_dict['batch_box_preds']
        if self.tta:
            batch_box_preds, data_dict = self.test_time_augment(
                data_dict,
                batch_box_preds
            )

        if self.model_cfg.POST_PROCESSING.get('GENERATE_RECALL', True):
            recall_dict = self.generate_recall_record(
                box_preds=batch_box_preds,
                recall_dict=recall_dict,
                data_dict=data_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        pred_dicts = {
            'pred_boxes': batch_box_preds.detach().cpu().numpy(),
            'pose': data_dict['pose'],
            'geo_trajectory': data_dict['geo_trajectory'],
        }

        return pred_dicts, recall_dict

    @staticmethod
    def test_time_augment(data_dict, boxes):
        raise NotImplementedError

    def generate_recall_record(self, box_preds, recall_dict,
                               data_dict=None, thresh_list=None):

        if 'gt_geo_trajectory' not in data_dict:
            return recall_dict

        traj = data_dict['geo_trajectory']
        gt_boxes = data_dict['gt_geo_query_boxes']
        gt_traj = data_dict['gt_geo_trajectory']
        state = np.array(data_dict['state'])
        mth = data_dict['matched']
        mth_tk = np.array(data_dict['matched_tracklet'])

        # initilize the return result
        if recall_dict.__len__() == 0:
            recall_dict = {
                'Box num': 0,
                'Track num': 0,
                'static': 0,
                'dynamic': 0
            }
            
            for cur_th in thresh_list:
                recall_dict.update({
                    # box level result
                    'Box level input %s' % str(cur_th): 0,
                    'Box level output %s' % str(cur_th): 0,
                    'Box level input (static) %s' % str(cur_th): 0,
                    'Box level output (static) %s' % str(cur_th): 0,
                    'Box level input (dynamic) %s' % str(cur_th): 0,
                    'Box level output (dynamic) %s' % str(cur_th): 0,
                    # track level result
                    'Track level input %s' % str(cur_th): 0,
                    'Track level output %s' % str(cur_th): 0,
                    'Track level input (static) %s' % str(cur_th): 0,
                    'Track level output (static) %s' % str(cur_th): 0,
                    'Track level input (dynamic) %s' % str(cur_th): 0,
                    'Track level output (dynamic) %s' % str(cur_th): 0
                })

        cur_gt = gt_boxes
        cur_gt_num = 0
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]
        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou_box_input = []
                iou_box_output = []
                iou_tk_input = []
                iou_tk_output = []

                for idx, per_gt_traj in enumerate(gt_traj):
                    # false positive tracks are removed for statistics
                    if mth_tk[idx] == False:
                        continue
                    mth_ind = torch.from_numpy(mth[idx])
                    cur_gt_num += mth_ind.sum().item()
                    
                    box_preds_rep = box_preds[idx].repeat(per_gt_traj.shape[0], 1)
                    if not self.model_cfg.POST_PROCESSING.get('CENTER_ALIGN', False):
                        box_preds_rep[:, 0:3] = torch.from_numpy(traj[idx][:, 0:3]).float().cuda()
                        box_preds_rep[:, 6] = torch.from_numpy(traj[idx][:, 6]).float().cuda()
                    else:
                        # set the center origin and heading angle as 0
                        traj[idx][:, [0, 1, 2, 6]] = 0.
                        per_gt_traj[idx][:, [0, 1, 2, 6]] = 0.

                    per_iou_box_in = iou3d_nms_utils.boxes_iou3d_gpu(
                        torch.from_numpy(traj[idx][:, 0:7]).float().cuda()[mth_ind],
                        torch.from_numpy(per_gt_traj[:, 0:7]).float().cuda()[mth_ind]
                    ).diag()

                    per_iou_box_out = iou3d_nms_utils.boxes_iou3d_gpu(
                        box_preds_rep[:, 0:7][mth_ind],
                        torch.from_numpy(per_gt_traj[:, 0:7]).float().cuda()[mth_ind]
                    ).diag()

                    # calculate for box level result
                    iou_box_input.append(per_iou_box_in)
                    iou_box_output.append(per_iou_box_out)

                    # calculate for track level result
                    obj_len = mth_ind.sum().item()
                    tk_input_num = (per_iou_box_in > thresh_list[0]).sum().item()
                    iou_tk_input.append(tk_input_num / obj_len)
                    tk_output_num = (per_iou_box_out > thresh_list[0]).sum().item()
                    iou_tk_output.append(tk_output_num / obj_len)

                    # statistics based on the motion state
                    recall_dict['%s' % state[idx]] += obj_len
                    for cur_th in thresh_list:
                        recall_dict['Box level input (%s) %s' % (state[idx], str(cur_th))] +=\
                            (per_iou_box_in > cur_th).sum().item()
                        recall_dict['Box level output (%s) %s' % (state[idx], str(cur_th))] +=\
                            (per_iou_box_out > cur_th).sum().item()

                if len(iou_box_output) > 0:
                    all_iou_box_in = torch.cat(iou_box_input)
                    all_iou_box_out = torch.cat(iou_box_output)

                    all_iou_tk_in = torch.Tensor(iou_tk_input)
                    all_iou_tk_out = torch.Tensor(iou_tk_output)

                    for i, state in enumerate(state[mth_tk]):
                        for cur_th in thresh_list:
                            recall_dict['Track level input (%s) %s' % (state, str(cur_th))] += \
                                all_iou_tk_in[i] > 0.7
                            recall_dict['Track level output (%s) %s' % (state, str(cur_th))] += \
                                all_iou_tk_out[i] > 0.7

                else:
                    all_iou_box_in = torch.Tensor([]).cuda()
                    all_iou_box_out = torch.Tensor([]).cuda()
                    all_iou_tk_in = torch.Tensor([]).cuda()
                    all_iou_tk_out = torch.Tensor([]).cuda()


            for cur_th in thresh_list:
                recall_dict['Box level input %s' % str(cur_th)] += \
                    (all_iou_box_in > cur_th).sum().item()
                recall_dict['Box level output %s' % str(cur_th)] += \
                    (all_iou_box_out > cur_th).sum().item()

                recall_dict['Track level input %s' % str(cur_th)] +=\
                    (all_iou_tk_in > cur_th).sum().item()
                recall_dict['Track level output %s' % str(cur_th)] +=\
                    (all_iou_tk_out > cur_th).sum().item()

            recall_dict['Box num'] += cur_gt_num
            recall_dict['Track num'] += np.where(mth_tk == True)[0].shape[0]

        return recall_dict

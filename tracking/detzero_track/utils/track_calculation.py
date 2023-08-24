from collections import defaultdict

import numpy as np
import torch

from detzero_track.models.tracking_modules.data_association import IoUBEV_dis_mat, IoU3D_dis_mat


def get_iou_mat_dict(gt_data, pred_data, class_names, distinguish_class=True, iou='bev'):
    """
    calculate the IoU matrix for whole sequence
    Args:
        gt_data: gt data
        pred_data: predict data
        class_names: target class tot calculate
        distinguish_class: IoU of diffrencet class would be 0 if True, False otherwise
        iou: the method of calculate IoU
    Returns:
        iou_mat_dict: output the dict of IoU matrix that index by frame id
    """
    iou_mat_dict = dict()
    for index, sample_idx in enumerate(list(gt_data.keys())):
        if sample_idx in pred_data.keys():
            track_boxes_lidar = pred_data[sample_idx]['boxes_lidar']
            track_len = len(track_boxes_lidar)
        else:
            track_len = 0

        annos = gt_data[sample_idx]['annos']
        name_len = len(gt_data[sample_idx]['annos']['name'])
        if name_len == 0 or track_len == 0:
            iou_mat = np.zeros((name_len, track_len), dtype=np.float32)
        else:
            name_mask = np.zeros_like(annos['name'], dtype=np.bool)
            for class_n in class_names:
                name_mask = name_mask | (annos['name'] == class_n)
            gt_boxes_lidar = annos['gt_boxes_lidar'][name_mask, :]
            if iou == 'bev':
                iou_mat = IoUBEV_dis_mat(torch.from_numpy(gt_boxes_lidar[:, :7]).float().cuda(), 
                                         torch.from_numpy(track_boxes_lidar[:, :7]).float().cuda())
            elif iou == '3d':
                iou_mat = IoU3D_dis_mat(torch.from_numpy(gt_boxes_lidar[:, :7]).float().cuda(), 
                                        torch.from_numpy(track_boxes_lidar[:, :7]).float().cuda())

            if distinguish_class:
                track_name = pred_data[sample_idx]['name']
                for gt_idx, gt_n in enumerate(annos['name'][name_mask]):
                    diff_mask =  (track_name != gt_n)
                    iou_mat[gt_idx, diff_mask] = 0.
        iou_mat_dict[sample_idx] = iou_mat

    return iou_mat_dict


def get_gt_id_data(gt_data, gt_keys, class_names):
    """
    convert gt data from index by frame to index by obj_id 
    Args:
        gt_data: raw gt data
        gt_keys: keep the attributes
        class_names: keep the class type
    Returns:
        gt_id_data: output new gt data index by obj_id
    """
    gt_id_data = dict()
    for sample_idx, item in gt_data.items():
        annos = item['annos']
        name_len = len(gt_data[sample_idx]['annos']['name'])
        if name_len == 0: continue
        name_mask = np.zeros_like(annos['name'], dtype=np.bool)
        for class_n in class_names:
            name_mask = name_mask | (annos['name'] == class_n)
        for idx, obj_id in enumerate(annos['obj_ids'][name_mask]):
            if obj_id not in gt_id_data.keys():
                gt_id_data[obj_id] = defaultdict(list)
            for key in gt_keys:
                gt_id_data[obj_id][key].append(annos[key][name_mask][idx])
            gt_id_data[obj_id]['sample_idx'].append(str(sample_idx))
            gt_id_data[obj_id]['iou_idx'].append(idx)

    return gt_id_data


def get_trajectory_similarity(track_a, track_b, iou_mat_dict,
                              iou_thresholds, least_len=0.):
    """
    """
    tk_a_frm_id_list = [int(x) for x in track_a['sample_idx']]
    tk_b_frm_id_list = [int(x) for x in track_b['sample_idx']]
        
    if tk_a_frm_id_list[0] > tk_b_frm_id_list[-1] or \
        tk_a_frm_id_list[-1] < tk_b_frm_id_list[0]:
        return -1, 0, 0
    
    similarity = 0
    match_count = 0
    same_frame_count = 0
    track_a_idx = 0
    track_b_idx = 0
    while track_a_idx < len(tk_a_frm_id_list) and \
        track_b_idx < len(tk_b_frm_id_list):

        if tk_a_frm_id_list[track_a_idx] == tk_b_frm_id_list[track_b_idx]:
            iou = iou_mat_dict[str(tk_a_frm_id_list[track_a_idx])]\
                [track_a['iou_idx'][track_a_idx], track_b['iou_idx'][track_b_idx]]
            similarity += iou
            if iou >= iou_thresholds[track_a['name'][track_a_idx]]:
                match_count += 1
            track_a_idx += 1
            track_b_idx += 1
            same_frame_count += 1
        else:
            if tk_a_frm_id_list[track_a_idx] < tk_b_frm_id_list[track_b_idx]:
                track_a_idx += 1
            else:
                track_b_idx += 1
    
    if match_count/len(tk_a_frm_id_list) >= least_len and match_count > 0:
        similarity = similarity/len(tk_a_frm_id_list)
    else:
        similarity = -1.
    
    return similarity, match_count, same_frame_count


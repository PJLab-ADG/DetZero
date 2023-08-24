import os
import torch
import pickle
from collections import defaultdict

import numpy as np

from detzero_utils.visualize_utils import LabelLUT, DataCollect, VisualizerGUI
from detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

from detzero_track.utils.data_utils import sequence_list_to_dict
from detzero_track.utils.transform_utils import transform_boxes3d


def sequence_visualize3d(color_attr, text_attr, **infos):
    data_collect = DataCollect(color_attr=color_attr,  # 'class', 'id
                               text_attr=text_attr,  # 'score', 'class', 'id'
                               show_text=True)
    data_collect.offline_process_infos(**infos)

    lut = LabelLUT()
    if len(data_collect.color_attr) == 0:
        lut_labels = {
            "gt": [0., 1., 0.],
            "pred": [0., 0., 1.],
        }
    else:
        lut_labels = {
            "pred_Vehicle": [0., 0., 1.0],
            "pred_Pedestrian": [1., 0., 0.],
            "pred_Cyclist": [0., 1.0, 0.0],
            "gt_Vehicle": [0.5, 1.0, 0.25],
            "gt_Pedestrian": [0.375, 0.375, 0.375],
            "gt_Cyclist": [0.11764706, 0.11764706, 1.],
        }
    for key, val in lut_labels.items():
        lut.add_label(key, key, val)
    if "id" in data_collect.color_attr:
        lut = None
    
    _detzero_vis = VisualizerGUI(fps=10)
    _detzero_vis.visualize_dataset(data_collect, prefix="frame id", lut=lut)


def load_waymo(pred_path, seq_name, pts_path, object_id=False, points_in_box=False, 
               load_gt=False, gt_path=None, ego=False):
    
    raw_pred_data = pickle.load(open(pred_path, "rb"))
    pred_data = sequence_list_to_dict(raw_pred_data)

    seq_name_list = list(pred_data.keys())
    if load_gt:
        gt_infos = pickle.load(open(gt_path, "rb"))
        gt_info_table = defaultdict(dict)
        for item in gt_infos:
            gt_info_table[item['sequence_name']][str(item['sample_idx'])] = item

    show_seq_n = seq_name
    pts_seq_path = os.path.join(pts_path, f"segment-{show_seq_n}")
    pred_data = pred_data[show_seq_n]
    if load_gt:
        gt_data = gt_info_table[show_seq_n]
  
    frame_id_list = sorted(list(pred_data.keys()), key=int)
    pts_list = list()
    pts_label_list = list()
    pred_list = list()
    gt_list = list()
    frame_ids_list = list()
    for i, frm_id in enumerate(frame_id_list):
        pts_path = pts_seq_path + '/' + ('0000'+frm_id)[-4:] + '.npy'
        pts = np.load(pts_path)
        only_pts = pts[:, :3]
        if not ego:
            pose = pred_data[frm_id]["pose"]
            only_pts = np.concatenate([only_pts, np.ones((pts.shape[0], 1))], axis=-1)
            only_pts = only_pts @ pose.T
        pts[:, :3] = only_pts[:, :3]

        pred_lidar = pred_data[frm_id]["boxes_lidar"]
        pred_mask = pred_data[frm_id]["score"] >= 0.        
        pred_boxes = pred_lidar if ego else transform_boxes3d(pred_lidar, pose)
        
        if load_gt:
            gt_mask = gt_data[frm_id]["annos"]["name"] != 'Sign'  
            gt_boxes = gt_data[frm_id]["annos"][f"gt_boxes_{'lidar' if ego else 'global'}"]

        if points_in_box:
            box_idxs = points_in_boxes_gpu(
                torch.from_numpy(pts[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(pred_boxes[pred_mask, :7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()
        else:
            box_idxs = np.zeros(pts.shape[0], dtype=np.int32) - 1

        dis = np.linalg.norm(pts[:, :2], axis=1)
        pts_mask = dis > 3
        new_pts = pts[pts_mask, :4]
        new_pts[:, 3] = np.tanh(new_pts[:, 3])
        if len(new_pts) == 0: continue

        frame_ids_list.append(frm_id)
        pts_list.append(new_pts[:, :])
        pts_label_list.append(box_idxs[:])
        
        pred_list.append({
            "bbox": pred_boxes[pred_mask, :7],
            "class": pred_data[frm_id]["name"][pred_mask],
            "score": pred_data[frm_id]["score"][pred_mask],
        })
        if object_id:
            pred_list[-1].update({"id": pred_data[frm_id]["obj_ids"][pred_mask]})

        if load_gt:
            gt_list.append({
                "bbox": gt_boxes[gt_mask, :7],
                "class": gt_data[frm_id]["annos"]["name"][gt_mask],
                "id": gt_data[frm_id]["annos"]["obj_ids"][gt_mask],
            })

    info = {
        "idx_names": frame_ids_list,
        "pts": pts_list,
        "pts_label": pts_label_list
    }

    info.update({"pred": pred_list})
    if load_gt:
        info.update({"gt": gt_list})
    
    return info

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, required=False, help='load pred data')
    parser.add_argument('--seq_name', type=str, default="13982731384839979987_1680_000_1700_000", help='the sequence name')
    parser.add_argument('--object_id', action='store_true', help='load object id')
    parser.add_argument('--color_attr', type=str, nargs='+', default=[], help='color attribute of 3d boxes')
    parser.add_argument('--text_attr', type=str, nargs='+', default=[], help='text attribute of 3d boxes')
    parser.add_argument('--ego', action='store_true', help='using ego coordinate')
    parser.add_argument('--points_in_box', action='store_true', help='colored points in box')
    parser.add_argument('--pts_path', type=str, default='../data/waymo/waymo_processed_data', help='the path of gt infos')
    parser.add_argument('--load_gt', action='store_true', help='loading gt data')
    parser.add_argument('--gt_path', type=str, default='../data/waymo/waymo_infos_val.pkl', help='the path of gt infos')
    args = parser.parse_args()

    info = load_waymo(
        pred_path=args.data_path, 
        seq_name=args.seq_name,
        pts_path=args.pts_path,
        object_id=args.object_id,
        points_in_box=args.points_in_box,
        load_gt=args.load_gt,
        gt_path=args.gt_path,
        ego=args.ego
    )
    sequence_visualize3d(args.color_attr, args.text_attr, **info)

import numpy as np

from detzero_track.utils.data_utils import frame_list_to_dict, tracklets_to_frames
from detzero_track.utils.track_calculation import get_iou_mat_dict, get_gt_id_data
from .data_association import GNN_assignment


def assign_track_target(input_data, iou_thresholds):
    """
    Function:
        Assign ground-truth data for object tracks

    Args:
    - det_data: dictionary containing detection data
    - tk_data: dictionary containing tracking data
    - gt_data: dictionary containing ground truth data
    - class_names: list of class names to consider
    - iou_thresholds: dictionary containing IOU thresholds for each class

    Returns:
    - dictionary containing labeled and unlabeled data
    """
    det_data, tk_data, gt_data = input_data[0], input_data[1], input_data[2]
    class_names = list(iou_thresholds.keys())

    # Convert data to desired format
    list_track_data = tracklets_to_frames({
        'reference': det_data,
        'source': tk_data
    })
    list_track_data = frame_list_to_dict(list_track_data)

    # Get IOU matrix dictionary
    list_gt_data = gt_data
    iou_mat_dict = get_iou_mat_dict(
        list_gt_data, list_track_data, class_names, True, 'bev')

    # Get ground truth data by ID
    gt_keys = ['gt_boxes_global', 'gt_boxes_lidar', 'name', 'obj_ids']            
    gt_data = get_gt_id_data(list_gt_data, gt_keys, class_names)
    gt_ids = list(gt_data.keys())
    tk_ids = list(tk_data.keys())

    # Initialize similarity and count matrices
    traj_similar_mat = np.zeros((len(gt_ids), len(tk_ids)), dtype=np.float32)
    traj_count_mat = np.zeros((len(gt_ids), len(tk_ids)), dtype=np.int)

    # Add IOU indices to tracking data
    for key, val in list_track_data.items():
        for iou_idx, obj_id in enumerate(val['obj_ids']):
            tk_data[obj_id]['iou_idx'].append(iou_idx)

    # Loop through frames and match ground truth data with tracking data
    frame_list = list(list_gt_data.keys())
    for idx, sample_idx in enumerate(frame_list):
        frame_gt_data = list_gt_data[sample_idx]
        frame_track_data = list_track_data[sample_idx]
        iou_mat = iou_mat_dict[sample_idx]

        for f_idx, gt_id in enumerate(frame_gt_data['annos']['obj_ids']):
            gt_name = frame_gt_data['annos']['name'][f_idx]
            if gt_name not in class_names: continue

            gt_id_idx = gt_ids.index(gt_id)
            sample_gt_idx = gt_data[gt_id]['sample_idx'].index(sample_idx)
            gt_idx = gt_data[gt_id]['iou_idx'][sample_gt_idx]

            for i, tk_id in enumerate(frame_track_data['obj_ids']):
                tk_id_idx = tk_ids.index(tk_id)
                track_name = frame_track_data['name'][i]
                simliarity = iou_mat[gt_idx, i]

                if gt_name == track_name and simliarity >= iou_thresholds[gt_name]:
                    traj_count_mat[gt_id_idx, tk_id_idx] += 1
                    traj_similar_mat[gt_id_idx, tk_id_idx] += simliarity

    # Calculate final similarity matrix and perform GNN assignment
    for gt_idx, gt_id in enumerate(gt_ids):
        gt_val = gt_data[gt_id]
        gt_len = len(gt_val['sample_idx'])

        for i, tk_id in enumerate(tk_ids):
            track_val = tk_data[tk_id]
            track_len = len(track_val['sample_idx'])
            final_simliarity = traj_similar_mat[gt_idx, i] / gt_len
            if traj_count_mat[gt_idx, i] <= 0:
                final_simliarity = -1.
            traj_similar_mat[gt_idx, i] = final_simliarity

    match, unmatch_gt, unmatch_track = GNN_assignment(1-traj_similar_mat)

    # Create labeled and unlabeled data dictionaries
    label_data = dict()
    unlabel_data = dict()

    # Add labeled data to dictionary
    for match_idx in range(len(match)):
        tk_id = tk_ids[match[match_idx, 1]]
        gt_id = gt_ids[match[match_idx, 0]]

        tk_data[tk_id]['iou'] = np.zeros(len(tk_data[tk_id]['sample_idx']), np.float32)
        inter_sample_idx = np.intersect1d(gt_data[gt_id]['sample_idx'], 
                                            tk_data[tk_id]['sample_idx'])
        for idx, sample_idx in enumerate(inter_sample_idx):
            sample_gt_idx = gt_data[gt_id]['sample_idx'].index(sample_idx)
            iou_gt_idx = gt_data[gt_id]['iou_idx'][sample_gt_idx]
            sample_tk_idx = list(tk_data[tk_id]['sample_idx']).index(sample_idx)
            iou_tk_idx = tk_data[tk_id]['iou_idx'][sample_tk_idx]

            match_iou = iou_mat_dict[sample_idx][iou_gt_idx, iou_tk_idx]
            tk_data[tk_id]['iou'][sample_tk_idx] = match_iou

        gt_data[gt_id].pop('iou_idx')
        for key in gt_data[gt_id].keys():
            gt_data[gt_id][key] = np.array(gt_data[gt_id][key])

        pos_diff = np.linalg.norm(gt_data[gt_id]['gt_boxes_global'][0, :2] - \
            gt_data[gt_id]['gt_boxes_global'][-1, :2], ord=2, axis=0)
        speed = np.linalg.norm(gt_data[gt_id]['gt_boxes_global'][:, 7:9], ord=2, axis=1)
        if speed.any() > 1 or pos_diff > 1:
            tk_data[tk_id]['state'] = 'dynamic'
        else:
            tk_data[tk_id]['state'] = 'static'

        tk_data[tk_id].pop('iou_idx')
        label_data[tk_id] = {
            'track': tk_data[tk_id],
            'gt': gt_data[gt_id]
        }

    # Add unlabeled data to dictionary
    for unmatch_tk_idx in unmatch_track:
        tk_id = tk_ids[unmatch_tk_idx]
        tk_data[tk_id].pop('iou_idx')

        unlabel_data[tk_id] = {
            'track': tk_data[tk_id]
        }
    return {'label':label_data, 'unlabel':unlabel_data}

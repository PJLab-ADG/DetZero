import os
import pickle
import argparse
import yaml
from pathlib import Path

import numpy as np
import torch

from detzero_utils.common_utils import create_logger
from detzero_utils.ops.iou3d_nms import iou3d_nms_utils


def generate_refine_boxes_iou(class_name, geo_path, pos_path, root_path, logger):
    """
    Function:
        Generate data infos used in CRM training
    Input:
        class_name: object category
        geo_path: GRM result for training split
        pos_path: PRM result for training split
        root_path: saved path of the generated IoU result
    """

    if 'train' not in geo_path or 'train' not in pos_path:
        raise ValueError('Please provide the refining results for training split.')

    geo_res = pickle.load(open(geo_path, 'rb'))
    pos_res = pickle.load(open(pos_path, 'rb'))

    data_info = {}
    seq_names = list(geo_res.keys())
    for seq in seq_names:
        data_info[seq] = {}

        obj_ids = geo_res[seq].keys()
        for obj_id in obj_ids:
            geo_pred = np.array(geo_res[seq][obj_id]['boxes_lidar']).reshape(-1, 7)
            pos_pred = np.array(pos_res[seq][obj_id]['boxes_global'])

            boxes_refine = pos_pred.copy()
            boxes_refine[:, 3:6] = geo_pred[:, 3:6]
            boxes_refine = torch.from_numpy(boxes_refine).cuda()

            boxes_gt = np.array(pos_res[seq][obj_id]['boxes_gt_global'])
            boxes_gt = torch.from_numpy(boxes_gt).cuda()
            
            iou = iou3d_nms_utils.boxes_iou3d_gpu(
                boxes_refine[:, 0:7],
                boxes_gt[:, 0:7]
            ).diag().cpu().numpy()

            data_info[seq].update({obj_id: iou})

    iou_path = os.path.join(root_path, '%s_iou_train.pkl' % class_name)
    with open(iou_path, 'wb') as f:
        pickle.dump(data_info, f)

    logger.info('The generated IoU label is saved at %s' % iou_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--class_name', type=str, default='Vehicle', help='class name')
    parser.add_argument('--geo_path', type=str, help='file path of geometry refine result')
    parser.add_argument('--pos_path', type=str, help='file path of position refine result')

    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()

    generate_refine_boxes_iou(
        args.class_name,
        args.geo_path,
        args.pos_path,
        os.path.join(ROOT_DIR, 'data/waymo/refining'),
        logger=create_logger()
    )

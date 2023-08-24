import os
import argparse
import yaml
from easydict import EasyDict
from pathlib import Path
import pickle

import numpy as np
import torch

from detzero_utils.common_utils import create_logger, multi_processing
from detzero_utils.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu_v2


class WaymoObjectDataPrepare():
    """
    Function:
        Prepare data infos used for refining module
    """
    def __init__(self, class_name, root_path=None, split='train', track_data_path=None,
                 enlarge_scale=1.1, crop_on_bev=False, workers=1, logger=None):
        
        self.class_name = class_name
        self.root_path = root_path
        self.split = split
        self.tk_data_path = track_data_path
        
        self.enlarge_scale = enlarge_scale
        self.crop_on_bev = crop_on_bev
        self.workers = workers
        self.logger = logger
        
        self.processed_infos = {}
        self.save_path = os.path.join(root_path, 'refining', class_name)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def init_infos_from_tracking(self):
        """
        Function:
            Load the results of upstream modules in the format of
            dict (sequence_id as keys), and corresponding infos
        """        
        self.logger.info('Loading generated object tracks for %s set.' % self.split)
        with open(self.tk_data_path, 'rb') as f:
            self.waymo_infos = pickle.load(f)
        self.logger.info('Total object tracks for Waymo dataset: %d.' % len(self.waymo_infos))

        self.logger.info('Start to convert object tracks into frame-level format.')
        seq_names = list(self.waymo_infos.keys())
        res = multi_processing(self.convert_track_into_frame, seq_names, self.workers, bar=True)
        self.processed_infos = dict(zip(seq_names, res))

        self.logger.info('Start to crop object points.')
        multi_processing(self.crop_object_points, seq_names, self.workers, bar=True)
        self.logger.info('Object data preparison finished.')

    def convert_track_into_frame(self, seq):
        output_dict = {}

        # train split: use gt to train the refining model
        # val split: use gt to calculate the recall metric
        if self.split in ['train', 'val']:
            # process the tracklets which have corresponding gt labels
            tracklet_ids = self.waymo_infos[seq]['label'].keys()
            for tk_id in tracklet_ids:
                tklet = self.waymo_infos[seq]['label'][tk_id]
                tk_infos = tklet['track']
                gt_infos = tklet['gt']
                
                # only process one class
                if self.class_name not in tk_infos['name']:
                    continue

                # read tracking infos
                obj_id = tk_infos['obj_ids']
                name = tk_infos['name']
                boxes = tk_infos['boxes_global']
                score = tk_infos['score']
                hit = tk_infos['hit']
                sample_idx = tk_infos['sample_idx']
                state = tk_infos['state']
                pose = tk_infos['pose']

                # read corresponding gt infos
                gt_obj_id = gt_infos['obj_ids']
                gt_name = gt_infos['name']
                gt_boxes = gt_infos['gt_boxes_global']
                gt_sample_idx = gt_infos['sample_idx']

                # set dummy id for non-matching frames
                dummy_gt_id = gt_obj_id[0]
                dummy_gt_name = gt_name[0]

                for idx, frm_id in enumerate(sample_idx):
                    frm_id = str(frm_id).zfill(4)
                    if frm_id not in output_dict:
                        tmp_info = {
                            'obj_id': [],
                            'name': [],
                            'boxes_global': [],
                            'score': [],
                            'hit': [],
                            'sample_idx': frm_id,
                            'pose': pose[idx],
                            'state': state,
                            'matched': [],
                            'matched_tracklet': [],
                            'gt_obj_id': [],
                            'gt_name': [],
                            'gt_boxes_global': []
                        }
                    else:
                        tmp_info = output_dict[frm_id]
                    
                    tmp_info['obj_id'].append(obj_id[idx])
                    tmp_info['name'].append(name[idx])
                    tmp_info['boxes_global'].append(boxes[idx][:7])
                    tmp_info['score'].append(score[idx])
                    tmp_info['hit'].append(hit[idx])
                    tmp_info['matched_tracklet'].append(True)

                    order_idx = np.where(gt_sample_idx == sample_idx[idx])[0]
                    if len(order_idx) > 0:
                        tmp_info['gt_obj_id'].append(gt_obj_id[order_idx[0]])
                        tmp_info['gt_name'].append(gt_name[order_idx[0]])
                        tmp_info['gt_boxes_global'].append(gt_boxes[order_idx[0]][:7])
                        tmp_info['matched'].append(True)
                    else:
                        tmp_info['gt_obj_id'].append(dummy_gt_id)
                        tmp_info['gt_name'].append(dummy_gt_name)
                        tmp_info['gt_boxes_global'].append(np.zeros_like(boxes[idx][:7]))
                        tmp_info['matched'].append(False)

                    output_dict[frm_id] = tmp_info

            
            # process the tracklets which have no matched gt labels
            fp_infos = self.waymo_infos[seq]['unlabel']
            tracklet_ids = fp_infos.keys()
            for tk_id in tracklet_ids:
                tk_infos = fp_infos[tk_id]['track']
                if self.class_name not in tk_infos['name']:
                    continue
                
                obj_id = tk_infos['obj_ids']
                name = tk_infos['name']
                boxes = tk_infos['boxes_global']
                score = tk_infos['score']
                hit = tk_infos['hit']
                sample_idx = tk_infos['sample_idx']
                state = tk_infos['state']
                pose = np.array(tk_infos['pose'])

                for idx, frm_id in enumerate(sample_idx):
                    frm_id = str(frm_id).zfill(4)
                    if frm_id not in output_dict:
                        tmp_info = {
                            'obj_id': [],
                            'name': [],
                            'boxes_global': [],
                            'score': [],
                            'hit': [],
                            'sample_idx': frm_id,
                            'pose': pose[idx],
                            'state': state,
                            'matched': [],
                            'matched_tracklet': [],
                            'gt_obj_id': [],
                            'gt_name': [],
                            'gt_boxes_global': []
                        }
                    else:
                        tmp_info = output_dict[frm_id]
                    
                    tmp_info['obj_id'].append(obj_id[idx])
                    tmp_info['name'].append(name[idx])
                    tmp_info['boxes_global'].append(boxes[idx][:7])
                    tmp_info['score'].append(score[idx])
                    tmp_info['hit'].append(hit[idx])
                    tmp_info['matched'].append(True)
                    tmp_info['matched_tracklet'].append(False)
                    tmp_info['gt_obj_id'].append(None)
                    tmp_info['gt_name'].append(None)
                    tmp_info['gt_boxes_global'].append(np.zeros_like(boxes[idx][:7]))

                    output_dict[frm_id] = tmp_info

        
        # test split: inference stage with no gt
        elif self.split == 'test':
            tracklet_ids = self.waymo_infos[seq].keys()
            for tk_id in tracklet_ids:
                tk_infos = self.waymo_infos[seq][tk_id]
                if self.class_name not in tk_infos['name']:
                    continue

                obj_id = tk_infos['obj_ids']
                name = tk_infos['name']
                boxes = tk_infos['boxes_global']
                score = tk_infos['score']
                hit = tk_infos['hit']
                sample_idx = tk_infos['sample_idx']
                state = tk_infos['state']
                pose = tk_infos['pose']

                for idx, frm_id in enumerate(sample_idx):
                    frm_id = str(frm_id).zfill(4)
                    if frm_id not in output_dict:
                        tmp_info = {
                            'obj_id': [],
                            'name': [],
                            'boxes_global': [],
                            'score': [],
                            'hit': [],
                            'sample_idx': frm_id,
                            'pose': pose[idx],
                            'state': state,
                            'matched': [],
                            'matched_tracklet': [],
                            'gt_boxes_global': []
                        }
                    else:
                        tmp_info = output_dict[frm_id]

                    tmp_info['obj_id'].append(obj_id[idx])
                    tmp_info['name'].append(name[idx])
                    tmp_info['boxes_global'].append(boxes[idx][:7])
                    tmp_info['score'].append(score[idx])
                    tmp_info['hit'].append(hit[idx])
                    tmp_info['matched_tracklet'].append(False)
                    tmp_info['matched'].append(True)
                    tmp_info['gt_boxes_global'].append(np.zeros_like(boxes[idx][:7]))

                    output_dict[frm_id] = tmp_info

        return output_dict

    def crop_object_points(self, seq):
        seq_info = self.processed_infos[seq]
        frame_ids = seq_info.keys()
        data_info = {}
        
        for frm_id in frame_ids:
            frm_info = seq_info[frm_id]
            for key in frm_info.keys():
                if key not in ['sample_idx', 'matched', 'matched_tracklet']:
                    frm_info[key] = np.array(frm_info[key])
            
            if len(frm_info['boxes_global']) > 0:
                # enlarge the boxes for cropping the points
                boxes_enlarge = frm_info['boxes_global'].copy()
                boxes_enlarge[:, 3:6] *= self.enlarge_scale
                if self.crop_on_bev:
                    boxes_enlarge[:, 5] = 100
                
                # load point cloud of current frame
                lidar_path = os.path.join(self.root_path,
                    'waymo_processed_data/segment-%s/%s.npy' % (seq, frm_id))
                pts = np.load(lidar_path)

                # transform the points into the global coordinate
                NLZ_flag = pts[:, 5]
                pts = pts[NLZ_flag == -1]
                pts_global = np.concatenate([pts[:, :3], np.ones((pts.shape[0], 1))], axis=-1)
                pts_global = pts_global @ frm_info['pose'].T
                pts = np.concatenate([pts_global[:, :3], np.tanh(pts[:, 3:4])], axis=1)

                obj_pts_mask = points_in_boxes_gpu_v2(
                    torch.from_numpy(pts[:, :3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(boxes_enlarge).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()
                obj_pts_mask = obj_pts_mask.astype(np.bool)

            for idx, obj_id in enumerate(frm_info['obj_id']):
                if obj_id not in data_info: 
                    obj_info_tmp = {
                        'sequence_name': seq,
                        'obj_id': obj_id,
                        'name': frm_info['name'][idx],
                        'boxes_global': [],
                        'score': [],
                        'sample_idx': [],
                        'hit': [],
                        'pose': [],
                        'state': frm_info['state'],
                        'matched': [],
                        'matched_tracklet': frm_info['matched_tracklet'][idx],
                        'pts': [],
                        'gt_boxes_global': []
                    }
                    
                    if frm_info['matched_tracklet'][idx]:
                        obj_info_tmp['gt_obj_id'] = frm_info['gt_obj_id'][idx]
                        obj_info_tmp['gt_name'] = frm_info['gt_name'][idx]
                    else:
                        obj_info_tmp['gt_obj_id'] = None
                        obj_info_tmp['gt_name'] = None
                else:
                    obj_info_tmp = data_info[obj_id]

                obj_info_tmp['boxes_global'].append(frm_info['boxes_global'][idx])
                obj_info_tmp['score'].append(frm_info['score'][idx])
                obj_info_tmp['sample_idx'].append(frm_id)
                obj_info_tmp['hit'].append(frm_info['hit'][idx])
                obj_info_tmp['pose'].append(frm_info['pose'])
                obj_info_tmp['matched'].append(frm_info['matched'][idx])
                obj_info_tmp['gt_boxes_global'].append(frm_info['gt_boxes_global'][idx])

                obj_pts = pts[obj_pts_mask[idx, :]]
                obj_info_tmp['pts'].append(obj_pts)

                data_info[obj_id] = obj_info_tmp

        # process the format and save object data
        for obj_id in data_info:
            obj_info = data_info[obj_id]
            keys = obj_info.keys()
            for key in keys:
                if key in ['obj_id', 'name', 'state', 'matched_tracklet',
                           'pts', 'sequence_name', 'gt_obj_id', 'gt_name']:
                    continue
                data_info[obj_id][key] = np.array(obj_info[key])

        save_path = os.path.join(self.save_path, '%s.pkl' % seq)
        with open(save_path, 'wb') as f:
            pickle.dump(data_info, f)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--enlarge_scale', type=float, default=1.1,
                        help='scale-up raito of object proposals')
    parser.add_argument('--crop_on_bev', type=bool, default=False,
                        help='whether to crop object points on bird-eye view')
    parser.add_argument('--split', type=str, default='val')
    parser.add_argument('--workers', type=int, default=1,
                        help='whether to use multi-process preparation')
    parser.add_argument('--track_data_path', type=str, default=None,
                        help='the generated tracking results pickle file')
    args = parser.parse_args()

    if args.split not in args.track_data_path:
        raise ValueError('The object tracks data does not match the dataset split.')

    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    logger = create_logger()

    class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
    
    for class_name in class_names:
        logger.info('Start to process %s data ...' % class_name)
        dataset = WaymoObjectDataPrepare(
            class_name=class_name,
            root_path=os.path.join(ROOT_DIR, 'data', 'waymo'),
            split=args.split,
            track_data_path=args.track_data_path,
            enlarge_scale=args.enlarge_scale,
            crop_on_bev=args.crop_on_bev,
            workers=args.workers,
            logger=logger
        )
        dataset.init_infos_from_tracking()

import os
import pickle
from collections import defaultdict
import argparse
from pathlib import Path

from tqdm import tqdm
from functools import partial
import concurrent.futures as futures
import numpy as np

from detzero_utils.common_utils import create_logger

from detzero_track.utils.transform_utils import transform_boxes3d
from detzero_track.utils.data_utils import sequence_list_to_dict, dict_to_sequence_list


def load_pkl(path):
    with open(path, 'rb') as f:
        info = pickle.load(f)
    return info

def save_pkl(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def combine_det(combine_data, drop_path):
    drop_data = load_pkl(drop_path)
    combine_data = sequence_list_to_dict(combine_data)
    seq_names = list(combine_data)
    
    for seq in seq_names:
        frames = list(combine_data[seq].keys())
        
        for frm in frames:
            for key in ['boxes_lidar', 'name', 'score']:
                combine_data[seq][frm][key] = np.concatenate([
                    combine_data[seq][frm][key],
                    drop_data[seq][frm][key]], axis=0)
    
    return dict_to_sequence_list(combine_data)


def convert_frame_format(track_data):
    """
    Function:
        convert track data from seq->obj_id dict format into frame-level list format
    Args:
        track_data: dict of track data {seq_n: {track_id: {xxxx}}}
    Returns:
        frame_res_list: list of frame-level result
    """
    frame_res_list = list()
    order_map = defaultdict(list)

    for tk_id, tk_info in track_data.items():
        sample_idx = tk_info['sample_idx']
        for i, sa_idx in enumerate(sample_idx):
            order_map[sa_idx].append([tk_id, i])

    frames = list(order_map.keys())
    for frm_id in frames:
        map_temp = np.stack(order_map[frm_id])
        obj_ids, orders = map_temp[:, 0], map_temp[:, 1]

        seq = track_data[obj_ids[0]]['sequence_name']
        pose = track_data[obj_ids[0]]['pose'][orders[0]]
        obj_num = len(obj_ids)

        boxes_lidar = np.zeros((obj_num, 7), dtype=np.float32)
        boxes_global = np.zeros((obj_num, 9), dtype=np.float32)
        score = np.zeros((obj_num), dtype=np.float32)
        name = np.full(obj_num, 'none', dtype=object)
        
        for i, obj_id in enumerate(obj_ids):
            idx = orders[i]
            
            if 'boxes_lidar' in track_data[obj_id]:
                boxes_lidar[i] = track_data[obj_id]['boxes_lidar'][idx]
            elif 'boxes_global' in track_data[obj_id]:
                boxes_global[i] = track_data[obj_id]['boxes_global'][idx]
                boxes_lidar[i] = transform_boxes3d(
                    boxes_global[i], pose, inverse=True).reshape(-1)
            
            score[i] = track_data[obj_id]['score'][idx]
            name[i] = track_data[obj_id]['name'][idx]
        
        frame_res_list.append({
            'sequence_name': seq,
            'frame_id': int(frm_id),
            'obj_ids': obj_ids,
            'name': name,
            'score': score,
            'boxes_lidar': boxes_lidar,
            'boxes_global': boxes_global,
            'pose': pose
        })
    
    return frame_res_list


def combine_final(root_path, class_names, logger, split='val', combine_conf_res=True,
                  combine_drop_path=None, track_save=True, frame_save=True):
    combine_dict = defaultdict(dict)
    root_path = os.path.join(root_path, 'result')
    
    for name in class_names:
        geo_path = os.path.join(root_path, '%s_geometry_%s.pkl' % (name, split))
        pos_path = os.path.join(root_path, '%s_position_%s.pkl' % (name, split))
        if not os.path.exists(geo_path) or not os.path.exists(pos_path):
            raise FileNotFoundError('Cannot find the input files.')
        
        geo_res = load_pkl(geo_path)
        pos_res = load_pkl(pos_path)
        
        if combine_conf_res:
            conf_path = os.path.join(root_path, '%s_confidence_%s.pkl' % (name, split))
            conf_res = load_pkl(conf_path)

        seq_names = list(pos_res.keys())
        print(len(seq_names))

        for seq in seq_names:
            obj_ids = pos_res[seq].keys()
            for obj in obj_ids:
                boxes_geo = np.concatenate(geo_res[seq][obj]['boxes_lidar'], axis=0)

                pos_res[seq][obj]['boxes_lidar'] = np.array(pos_res[seq][obj]['boxes_lidar'])
                pos_res[seq][obj]['boxes_lidar'][:, 3:6] = boxes_geo[:, 3:6]
                if combine_conf_res:
                    pos_res[seq][obj]['score'] = conf_res[seq][obj]['new_score']

                pos_res[seq][obj]['sample_idx'] = \
                    np.array([str(x) for x in pos_res[seq][obj]['frame_id']])
                
                combine_dict[seq][obj] = pos_res[seq][obj]

        if track_save:
            save_path = os.path.join(root_path, '%s_final.pkl' % name)
            save_pkl(combine_dict, save_path)
            logger.info('Track level final result is saved at %s' % save_path)

        if frame_save:
            logger.info('Start to convert track level result into frame level')

            final_res = list()    
            seq_data = [combine_dict[x] for x in seq_names]
            converter = partial(convert_frame_format)
            
            with futures.ProcessPoolExecutor(max_workers=1) as executor:
                thread_bar = tqdm(executor.map(
                    converter,
                    seq_data,
                    chunksize=1),
                    total=len(seq_names), ascii=True, ncols=140)
                
            for idx, processed_data in enumerate(thread_bar):
                final_res.extend(processed_data)

            # not combine dropped objects when used as auto labels
            if combine_drop_path is not None:
                logger.info('Start to combine the dropped objects from %s' % combine_drop_path)
                final_res = combine_det(final_res, combine_drop_path)

            save_path = os.path.join(root_path, '%s_final_frame.pkl' % name)
            save_pkl(final_res, save_path)
            logger.info('Frame level final result is saved at %s' % save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--num', type=int, default=None, help='specify the config for training')
    parser.add_argument('--path', type=str, default=None, help='specify the config for training')
    parser.add_argument('--split', type=str, default='val',
                        help='specify the target split for processing')

    parser.add_argument('--combine_conf_res', action='store_true', default=False,
                        help='combine the confidence refining results together')
    parser.add_argument('--combine_drop_path', type=str, default=None,
                        help='determine the dropped results at tracking module')

    parser.add_argument('--track_save', action='store_true', default=True,
                        help='save the combined result as original object track dict')
    parser.add_argument('--frame_save', action='store_true', default=True,
                        help='save the combined result as frame level list')
    
    args = parser.parse_args()

    ROOT_DIR = (Path(__file__).resolve().parent / '../').resolve()
    logger = create_logger()
    
    combine_final(
        root_path=os.path.join(ROOT_DIR, 'data/waymo/refining'),
        class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
        logger=logger,
        split=args.split,
        combine_conf_res=args.combine_conf_res,
        combine_drop_path=args.combine_drop_path,
        track_save=args.track_save,
        frame_save=args.frame_save,
    )

import copy
import numpy as np

from collections import defaultdict
from .transform_utils import transform_boxes3d


def frame_list_to_dict(data):
    new_data = defaultdict(dict)
    for item in data:
        new_data[str(item['sample_idx'])] = item
    return new_data


def sequence_list_to_dict(data):
    new_data = {}
    for item in data:
        sample_idx = str(item['sample_idx']) if 'sample_idx' in item.keys() else str(item['frame_id'])
        if item['sequence_name'] not in new_data:
            new_data[item['sequence_name']] = {}
        new_data[item['sequence_name']][sample_idx] = item
    return new_data


def dict_to_sequence_list(data):
    new_data = list()
    for seq_n in data.keys():
        for frame_id in data[seq_n].keys():
            new_data.append(data[seq_n][frame_id])
    return new_data


def tracklets_to_frames(data_dict):
    source_data = data_dict['source']
    reference_data = data_dict['reference']

    obejct_data_list = list()
    frame_object_dict = defaultdict(set)
    for obj_id, obj_data in source_data.items():
        sample_idx = obj_data['sample_idx']
        for sa_idx in sample_idx:
            frame_object_dict[sa_idx].add(obj_id)

    frame_list = list(reference_data.keys())
    for _, frm_id in enumerate(frame_list):
        seq = reference_data[frm_id]['sequence_name']
        pose = reference_data[frm_id]['pose']
        object_ids = np.array(sorted(list(frame_object_dict[frm_id])))
        
        obj_num = len(object_ids)
        boxes_lidar = np.zeros((obj_num, 7), dtype=np.float32)
        boxes_global = np.zeros((obj_num, 7), dtype=np.float32)
        score = np.zeros(obj_num, dtype=np.float32)
        name = np.full(obj_num, None, dtype=object)

        for idx, obj_id in enumerate(object_ids):
            object_data = source_data[obj_id]
            index = np.where(object_data['sample_idx'] == frm_id)[0][0]
            if 'boxes_lidar' in object_data.keys():
                boxes_lidar[idx] = copy.deepcopy(object_data['boxes_lidar'][[index], :7])
            else:
                boxes_global[[idx]] = object_data['boxes_global'][[index], :7]
                boxes_lidar[idx] = transform_boxes3d(boxes_global[[idx]], pose, inverse=True).reshape(-1)
            score[idx] = object_data['score'][index]
            name[idx] = object_data['name'][index]

        obejct_data_list.append({
            'sequence_name': seq,
            'sample_idx': frm_id,
            'obj_ids': object_ids,
            'name': name,
            'boxes_lidar': boxes_lidar,
            'score': score,
            'pose': pose
        })
    return obejct_data_list


def frames_to_tracklets(data_dict, class_names=['Vehicle', 'Pedestrian', 'Cyclist']):
    source_data = data_dict['source']
    obj_id_data = dict()
    keep_key_list = ['sample_idx', 'pose', 'sequence_name', 'timestamp']
    for sample_idx, frames_data in source_data.items():
        pose = frames_data[sample_idx]['pose']
        name_len = len(frames_data[sample_idx]['name'])
        if name_len == 0: continue

        name_mask = np.zeros_like(frames_data[sample_idx]['name'], dtype=np.bool)
        for class_n in class_names:
            name_mask = name_mask | (frames_data[sample_idx]['name'] == class_n)

        for idx, obj_id in enumerate(item['obj_ids'][name_mask]):
            if obj_id not in obj_id_data.keys():
                obj_id_data[obj_id] = defaultdict(list)
            for key in list(item.keys()):
                if key in keep_key_list:
                    continue
                obj_id_data[obj_id][key].append(item[key][name_mask][idx])

            obj_id_data[obj_id]['sample_idx'].append(str(sample_idx))
            obj_id_data[obj_id]['pose'].append(pose)

    for obj_id, item in obj_id_data.items():
        for k, v in item.items():
            obj_id_data[obj_id][k] = np.array(obj_id_data[obj_id][k])
    return obj_id_data

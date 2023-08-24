import copy

import numpy as np

from detzero_utils import common_utils
from detzero_refine.utils.data_utils import rotate_yaw


def augment_full_track(pts, traj, init_pts, init_box, traj_gt):
    init_sa_num = len(init_box)

    # Flip along X-axis
    if np.random.random() < 0.5:
        pts[:, 1] = -pts[:, 1]
        for i in range(init_sa_num):
            init_pts[i][:, 1] = -init_pts[i][:, 1]

    # Flip along Y-axis
    if np.random.random() < 0.5:
        pts[:, 0] = -pts[:, 0]
        for i in range(init_sa_num):
            init_pts[i][:, 0] = -init_pts[i][:, 0]

    # Rotate along Z-axis
    if np.random.random() < 0.5:
        angle = np.random.uniform(-np.pi/2, np.pi/2)
        pts[:, :3] = common_utils.rotate_points_along_z(
            pts[:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]
        for i in range(init_sa_num):
            init_pts[i][:, :3] = common_utils.rotate_points_along_z(
                init_pts[i][:, :3][np.newaxis, :, :],
                np.array([angle])
            )[0]

    # Scaling the size
    scale_range = [0.9, 1.1]
    if np.random.random() < 0.5:
        factor = np.random.uniform(scale_range[0], scale_range[1])
        pts[:, :3] *= factor
        pts[:, 4:] *= factor
        for i in range(init_sa_num):
            init_pts[i][:, :3] *= factor

        traj[:, 3:6] *= factor
        init_box[:, 3:6] *= factor
        traj_gt[:, 3:6] *= factor
    
    return pts, traj, init_pts, init_box, traj_gt



def coords_transform_one_box(pts, box, inverse=False):
    center_point = box[:3]
    if not inverse:
        pts[:, :3] = pts[:, :3] - center_point
        pts[:, :3] = pts[:, :3] @ rotate_yaw(box[6]).T
    else:
        pts[:, :3] = pts[:, :3] @ rotate_yaw(-box[6]).T
        pts[:, :3] = pts[:, :3] + center_point
    return pts


def augment_single_box(pts, traj=None):
    if traj is None:
        for one_obj_pts in pts:
            if np.random.random() < 0.5:
                one_obj_pts[:, 1] = -one_obj_pts[:, 1]
        return pts
    for idx in range(len(pts)):
        if np.random.random() < 0.5:
            pts[idx] = coords_transform_one_box(pts[idx], traj[idx])
            pts[idx][:, 1] = -pts[idx][:, 1]
            pts[idx] = coords_transform_one_box(pts[idx], traj[idx], inverse=True)
        
    return pts


def test_time_augment(data_dict):
    data_dict_tta = {}
    data_dict_tta['tta_original'] = data_dict

    #** Flip along X-axis **#
    data_dict_flip_x = copy.deepcopy(data_dict)
    data_dict_flip_x['geo_memory_points'][:, 1] = -data_dict_flip_x['geo_memory_points'][:, 1]
    for i in range(len(data_dict_flip_x['geo_query_points'])):
        data_dict_flip_x['geo_query_points'][i][:, 1] = -data_dict_flip_x['geo_query_points'][i][:, 1]
    data_dict_tta['tta_flip_x'] = data_dict_flip_x
    
    #** Flip along Y-axis **#
    data_dict_flip_y = copy.deepcopy(data_dict)
    data_dict_flip_y['geo_memory_points'][:, 0] = -data_dict_flip_y['geo_memory_points'][:, 0]
    for i in range(len(data_dict_flip_y['geo_query_points'])):
        data_dict_flip_y['geo_query_points'][i][:, 0] = -data_dict_flip_y['geo_query_points'][i][:, 0]
    data_dict_tta['tta_flip_y'] = data_dict_flip_y

    #** Flip along X&Y-axis **#
    data_dict_flip_xy = copy.deepcopy(data_dict)
    data_dict_flip_xy['geo_memory_points'][:, 0] = -data_dict_flip_xy['geo_memory_points'][:, 0]
    data_dict_flip_xy['geo_memory_points'][:, 1] = -data_dict_flip_xy['geo_memory_points'][:, 1]
    for i in range(len(data_dict_flip_xy['geo_query_points'])):
        data_dict_flip_xy['geo_query_points'][i][:, 0] = -data_dict_flip_xy['geo_query_points'][i][:, 0]
        data_dict_flip_xy['geo_query_points'][i][:, 1] = -data_dict_flip_xy['geo_query_points'][i][:, 1]
    data_dict_tta['tta_flip_xy'] = data_dict_flip_xy

    scale_factor = [0.9, 0.95, 1.05, 1.1]
    for factor in scale_factor:
        if factor == 1.: continue
        data_dict_scale = copy.deepcopy(data_dict)
        data_dict_scale['geo_memory_points'][:, :3] *= factor
        data_dict_scale['geo_memory_points'][:, 4:] *= factor
        data_dict_scale['geo_query_points'][:, 3:6] *= factor
        for i in range(len(data_dict_scale['geo_query_points'])):
            data_dict_scale['geo_query_points'][i][:, :3] *= factor

        key_name = 'tta_scale_%s' % str(factor)
        data_dict_tta[key_name] = data_dict_scale

    #** Rotation **#
    rot_angle = [-0.78539816, 0.78539816]
    for angle in rot_angle:
        if angle == 0.: continue
        data_dict_rot = copy.deepcopy(data_dict)
        data_dict_rot['geo_memory_points'][:, :3] = common_utils.rotate_points_along_z(
            data_dict_rot['geo_memory_points'][:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]
        for i in range(len(data_dict_rot['geo_query_points'])):
            data_dict_rot['geo_query_points'][i][:, :3] = common_utils.rotate_points_along_z(
                data_dict_rot['geo_query_points'][i][:, :3][np.newaxis, :, :],
                np.array([angle])
            )[0]
        key_name = 'tta_rot_%s' % str(angle)
        data_dict_tta[key_name] = data_dict_rot

    return data_dict_tta

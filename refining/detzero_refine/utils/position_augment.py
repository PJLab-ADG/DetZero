import copy

import numpy as np

from detzero_utils import common_utils
from .data_utils import limit_heading_range


def augment_full_track(self, local_pts, global_pts, traj, traj_gt):
    # Flip along X-axis
    if np.random.random() < 0.5:
        local_pts[..., 1] = -local_pts[..., 1]
        global_pts[..., 1] = -global_pts[..., 1]
        traj[:, 1] = -traj[:, 1]
        traj[:, 6] = -traj[:, 6]
        traj_gt[:, 1] = -traj_gt[:, 1]
        traj_gt[:, 6] = -traj_gt[:, 6]

    # Flip along Y-axis
    if np.random.random() < 0.5:
        local_pts[..., 0] = -local_pts[..., 0]
        global_pts[..., 0] = -global_pts[..., 0]
        traj[:, 0] = -traj[:, 0]
        # please check the degree coordinate
        traj[:, 6] = -(traj[:, 6] + np.pi)
        traj_gt[:, 0] = -traj_gt[:, 0]
        traj_gt[:, 6] = -(traj_gt[:, 6] + np.pi)

    # Rotate along Z-axis
    if np.random.random() < 0.5:
        angle = np.random.uniform(-np.pi, np.pi)
        local_pts[:, :3] = common_utils.rotate_points_along_z(
            local_pts[:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]
        
        global_pts[:, :3] = common_utils.rotate_points_along_z(
            global_pts[:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]

        # rotate the centers of all boxes
        traj[:, :3] = common_utils.rotate_points_along_z(
            traj[:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]
        
        # rotate the centers of gt trajectory
        traj_gt[:, :3] = common_utils.rotate_points_along_z(
            traj_gt[:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]

        # add the rotated angle to the trajectory and gt trajectory
        traj[:, 6] += angle
        traj_gt[:, 6] += angle


    # Scaling
    scale_range = [0.85, 1.15]
    if np.random.random() < 0.5:
        factor = np.random.uniform(scale_range[0], scale_range[1])
        # TODO: take care with the intensity and score
        local_pts[:, :3] *= factor
        local_pts[:, 4:-1] *= factor

        global_pts[:, :3] *= factor
        global_pts[:, 4:-1] *= factor

        traj[:, 0:6] *= factor
        traj_gt[:, 0:6] *= factor
    
    # # random shift
    # pts[:, :3] *= scale_factor
    # # random shift
    # pts[:, :3] += trans_factor

    # After all the data augmentation, we limit the heading range
    traj[:, 6] = limit_heading_range(traj[:, 6])
    traj_gt[:, 6] = limit_heading_range(traj_gt[:, 6])

    return local_pts, global_pts, traj, traj_gt


def test_time_augment(data_dict):
    data_dict_tta = {}
    data_dict_tta['tta_original'] = data_dict

    #** Flip along X-axis **#
    data_dict_flip_x = copy.deepcopy(data_dict)
    data_dict_flip_x['input_pts_data'][..., 1] = -data_dict_flip_x['input_pts_data'][..., 1]
    data_dict_flip_x['trajectory'][:, 1] = -data_dict_flip_x['trajectory'][:, 1]
    data_dict_flip_x['trajectory'][:, 6] = -data_dict_flip_x['trajectory'][:, 6]
    data_dict_tta['tta_flip_x'] = data_dict_flip_x
    
    #** Flip along Y-axis **#
    data_dict_flip_y = copy.deepcopy(data_dict)
    data_dict_flip_y['input_pts_data'][..., 0] = -data_dict_flip_y['input_pts_data'][..., 0]
    data_dict_flip_y['trajectory'][:, 0] = -data_dict_flip_y['trajectory'][:, 0]
    data_dict_flip_y['trajectory'][:, 6] = -(data_dict_flip_y['trajectory'][:, 6] + np.pi)
    data_dict_tta['tta_flip_y'] = data_dict_flip_y

    #** Flip along X&Y-axis **#
    data_dict_flip_xy = copy.deepcopy(data_dict)
    data_dict_flip_xy['input_pts_data'][..., 0] = -data_dict_flip_xy['input_pts_data'][..., 0]
    data_dict_flip_xy['input_pts_data'][..., 1] = -data_dict_flip_xy['input_pts_data'][..., 1]
    data_dict_flip_xy['trajectory'][:, 0] = -data_dict_flip_xy['trajectory'][:, 0]
    data_dict_flip_xy['trajectory'][:, 1] = -data_dict_flip_xy['trajectory'][:, 1]
    data_dict_flip_xy['trajectory'][:, 6] = data_dict_flip_xy['trajectory'][:, 6] - np.pi
    data_dict_tta['tta_flip_xy'] = data_dict_flip_xy

    scale_factor = [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]
    for factor in scale_factor:
        if factor == 1.: continue
        data_dict_scale = copy.deepcopy(data_dict)
        data_dict_scale['input_pts_data'][..., :3] *= factor
        if self.use_scores:
            data_dict_scale['input_pts_data'][..., 4:-1] *= factor
        else:
            data_dict_scale['input_pts_data'][..., 4:] *= factor
        data_dict_scale['trajectory'][:, :3] *= factor

        key_name = 'tta_scale_%s' % str(factor)
        data_dict_tta[key_name] = data_dict_scale

    #** Rotation **#
    rot_angle = [-0.39365818, -0.78539816, -1.17809724, -2.74889357, 0.39365818, 0.78539816, 1.17809724, 2.74889357]
    for angle in rot_angle:
        if angle == 0.: continue
        data_dict_rot = copy.deepcopy(data_dict)
        data_dict_rot["input_pts_data"][..., :3] = common_utils.rotate_points_along_z(
            data_dict_rot["input_pts_data"][..., :3],
            np.array([angle]).repeat(data_dict_rot["input_pts_data"].shape[0], 0)
        )

        data_dict_rot['trajectory'][:, :3] = common_utils.rotate_points_along_z(
            data_dict_rot['trajectory'][:, :3][np.newaxis, :, :],
            np.array([angle])
        )[0]

        data_dict_rot['trajectory'][:, 6] += angle

        key_name = 'tta_rot_%s' % str(angle)
        data_dict_tta[key_name] = data_dict_rot

    # limit heading range into [-pi, pi) after all the operations
    for key in data_dict_tta:
        data_dict_tta[key]['trajectory'][:, 6] = limit_heading_range(
            data_dict_tta[key]['trajectory'][:, 6])
    
    return data_dict_tta



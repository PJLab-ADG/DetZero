import numpy as np

from detzero_utils import common_utils


def random_flip_along_x(gt_boxes, points, return_enable_xy=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    if return_enable_xy:
        return gt_boxes, points, int(enable)
    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, return_enable_xy=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]
    if return_enable_xy:
        return gt_boxes, points, int(enable)
    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, return_rotate_noise=False):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
    Returns:
    """
    noise_rotation = np.random.uniform(rot_range[0], rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :], np.array([noise_rotation]))[0]
    gt_boxes[:, 0:3] = common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[np.newaxis, :, :],
            np.array([noise_rotation])
        )[0][:, 0:2]
    if return_rotate_noise:
        return gt_boxes, points, noise_rotation
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, return_scale_noise=False):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] *= noise_scale
    if return_scale_noise:
        return gt_boxes, points, noise_scale
    return gt_boxes, points


def global_translation(gt_boxes, points, std, return_std_noise=False):
    x_trans = np.random.randn(1)*std
    y_trans = np.random.randn(1)*std
    z_trans = np.random.randn(1)*std

    points[:, 0] += x_trans
    points[:, 1] += y_trans
    points[:, 2] += z_trans

    gt_boxes[:, 0] += x_trans
    gt_boxes[:, 1] += y_trans
    gt_boxes[:, 2] += z_trans
    if return_std_noise:
        return gt_boxes, points, np.array([x_trans, y_trans, z_trans]).T
    return gt_boxes, points

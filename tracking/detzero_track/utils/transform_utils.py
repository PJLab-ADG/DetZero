import numpy as np


def yaw_filter(yaw):
    """
    filter the heading into -pi ~ pi
    Args:
        yaw: np.ndarray or float, raw heading
    Returns:
        yaw: np.ndarray or float, filtered heading
    """
    pi2 = np.pi * 2

    if isinstance(yaw, np.ndarray):
        mask = np.abs(yaw) >= pi2
        yaw[mask] = yaw[mask] - np.floor(yaw[mask]/pi2)*pi2
        yaw[yaw > np.pi] -= pi2
        yaw[yaw <= -np.pi] += pi2
    else:
        if np.abs(yaw) >= pi2:
            yaw = yaw - np.floor(yaw/pi2)*pi2
            if yaw > np.pi: yaw -= pi2
            if yaw <= -np.pi: yaw += pi2

    return yaw


def get_inverse_transform_mat(src_pose):
    """ 
    Args:
        src_pose: 4*4 transform pose include rotate matrix and translation
    Returns:
        reverse_pose: 4*4 inverse of transform pose
    """
    reverse_pose = np.zeros((4, 4), dtype=np.float32)
    reverse_pose[:3, :3] = src_pose[:3, :3].T
    reverse_pose[:3, 3:] = -(src_pose[:3, :3].T @ src_pose[:3, 3:])
    reverse_pose[3, 3] = 1

    return reverse_pose


def transform_boxes3d(boxes, pose, inverse=False):
    """
    Args:
        boxes: N*7 x,y,z,dx,dy,dz,heading
        pose: 4*4 transform pose include rotate matrix and translation
        inverse: using inverse of transform pose if True, Fasle otherwise
    Returns:
        transformed_boxes: N*7 x,y,z,dx,dy,dz,heading
    """
    center = boxes[:, :3]
    center = np.concatenate([center, np.ones((center.shape[0], 1))], axis=-1)
    if inverse:
        pose = get_inverse_transform_mat(pose)
    center = center @ pose.T
    heading = yaw_filter(boxes[:, [6]] + np.arctan2(pose[1, 0], pose[0, 0]))

    return np.concatenate([center[:, :3], boxes[:, 3:6], heading], axis=-1)

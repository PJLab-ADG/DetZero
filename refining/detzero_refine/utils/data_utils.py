import random

import numpy as np


def rotate_yaw(yaw):
    return np.array([[np.cos(yaw), np.sin(yaw), 0],
                    [-np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]], dtype=np.float32)


def sample_points(pts, sample_num=4096, replace=False):
    pts_num, dim = pts.shape
    if pts_num >= sample_num:
        sample_idx = random.sample(range(0, pts_num), sample_num)
        sample_idx.sort()
        pts = pts[sample_idx]
        pts_num = sample_num

    else:
        if replace and pts_num > 0:
            sample_idx = np.arange(pts_num)
            sample_idx = np.tile(sample_idx, np.int(sample_num)//np.int(pts_num) + 1)[:sample_num]
            pts = pts[sample_idx]

        else:
            zeros = np.zeros((sample_num-pts_num, dim), dtype=np.float32)
            pts = np.concatenate((pts, zeros), axis=0)

    return pts


def limit_heading_range(angle):
    while (angle >= np.pi).sum() > 0:
        mask = (angle >= np.pi).nonzero()[0]
        angle[mask] -= 2*np.pi

    while (angle < -np.pi).sum() > 0:
        mask = (angle < -np.pi).nonzero()[0]
        angle[mask] += 2*np.pi

    return angle


def world_to_lidar(boxes, poses):
    boxes = np.stack(boxes, axis=0)
    poses = np.stack(poses, axis=0)
    r_t = np.linalg.inv(poses)
    num_box = len(boxes)
    centers = boxes[:, :3]
    heading = boxes[:, 6]
    centers = np.concatenate([centers, np.ones((num_box, 1))], axis=-1)
    centers = np.einsum('ijk,ikm->ijm', centers[:, None, :], r_t.transpose(0,2,1)).reshape(num_box, -1)
    heading = boxes[:, 6] + np.arctan2(r_t[:, 1, 0], r_t[:, 0, 0])
    box_lidar = np.concatenate([centers[:, :3], boxes[:, 3:6], heading[:, None]], axis=-1)
    return box_lidar


def local_coords_transform(pts, traj):
    """
    Function:
        Transform the points to the local box coordinate
    """
    traj_len = len(traj)
    for i in range(traj_len):
        pts[i][:, :3] = pts[i][:, :3] - traj[i][:3]
        pts[i][:, :3] = pts[i][:, :3] @ rotate_yaw(traj[i][6]).T

    return pts


def init_coords_transform(init_box, pts, traj=None, traj_gt=None):
    """
    Function:
        Transform the data to the init_box coordinate
    """
    # limit heading into [-pi, pi)
    init_box[6] = limit_heading_range(init_box[[6]])[0]

    # transform the lidar points to the new coordinate
    for i in range(len(pts)):
        pts[i][:, :3] -= init_box[:3]
        pts[i][:, :3] = pts[i][:, :3] @ rotate_yaw(init_box[6]).T

    # transform the trajectory to the new coordinate
    if traj is not None:
        traj[:, 6] = limit_heading_range(traj[:, 6])
        
        traj[:, :3] -= init_box[:3]
        traj[:, :3] = traj[:, :3] @ rotate_yaw(init_box[6]).T
        traj[:, 6] -= init_box[6]

        # limit heading into [-pi, pi)
        traj[:, 6] = limit_heading_range(traj[:, 6])

    # transform the gt trajectory to the new coordinate
    if traj_gt is not None:
        traj_gt[:, 6] = limit_heading_range(traj_gt[:, 6])
    
        traj_gt[:, :3] -= init_box[:3]
        traj_gt[:, :3] = traj_gt[:, :3] @ rotate_yaw(init_box[6]).T
        traj_gt[:, 6] -= init_box[6]

        traj_gt[:, 6] = limit_heading_range(traj_gt[:, 6])
    
    return init_box, pts, traj, traj_gt


def box_coords_transform(traj, init_box):
    traj[:, :3] = traj[:, :3] @ np.linalg.inv(rotate_yaw(init_box[6]).T)
    traj[:, :3] += init_box[:3]
    
    traj[:, 6] += init_box[6]
    traj[:, 6] = limit_heading_range(traj[:, 6])

    return traj


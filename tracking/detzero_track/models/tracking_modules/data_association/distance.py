import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from detzero_utils.ops.iou3d_nms import iou3d_nms_cuda
from detzero_utils.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu, boxes_giou3d_gpu


def GNN_assignment(cost_matrix, threshold=1.):
    """
    Args:
        cost_matrix: (N*M) range from 0~1, 0 represents similar, 1 represents not similar
        threshlod: cost_matrix[i][j] > threshlod would be not considered
    Returns:
        mathed, unmatched1, unmatched2
    """
    N, M = cost_matrix.shape
    if N == 0 or M == 0:
        return np.zeros((0, 2), dtype=np.int), np.arange(N), np.arange(M)

    # give a large num to avoid unmatched affect the global optim
    cost_matrix[cost_matrix >= threshold] = 5000.
    row_idx, col_idx = linear_sum_assignment(cost_matrix)

    matched_list = list()
    for idx in range(len(row_idx)):
        if cost_matrix[row_idx[idx], col_idx[idx]] < threshold:
            matched_list.append(np.array((row_idx[idx], col_idx[idx])))

    if len(matched_list):
        matched_list = np.array(matched_list)
    else: matched_list = np.empty((0, 2), dtype=np.int)

    unmatched1_list = list()
    unmatched2_list = list()
    for idx in range(N):
        if idx not in matched_list[:, 0]: unmatched1_list.append(idx)
    for idx in range(M):
        if idx not in matched_list[:, 1]: unmatched2_list.append(idx)

    return matched_list, np.array(unmatched1_list, dtype=np.int), np.array(unmatched2_list, dtype=np.int)


def bev_overlap_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]
    # bev overlap
    if N == 0 or M == 0:
        overlaps_bev = np.zeros((N, M), dtype=np.float32)
    else:
        overlaps_bev = torch.zeros((boxes_a.shape[0], boxes_b.shape[0])).cuda()
        iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)
        overlaps_bev = overlaps_bev.cpu().numpy()

    return overlaps_bev


def IoU2D_dis_mat(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N*4) x, y, w, h
        boxes_b: (M*4) x, y, w, h
    Returns:
        iou matrix: (N*M)
    """

    assert boxes_a.shape[1] == boxes_b.shape[1] == 4
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    boxes_a_x1y1 = boxes_a[:, :2] - boxes_a[:, 2:]*0.5
    boxes_a_x2y2 = boxes_a[:, :2] + boxes_a[:, 2:]*0.5

    boxes_b_x1y1 = boxes_b[:, :2] - boxes_b[:, 2:]*0.5
    boxes_b_x2y2 = boxes_b[:, :2] + boxes_b[:, 2:]*0.5

    max_x1y1 = np.maximum(boxes_a_x1y1[:, np.newaxis, :].repeat(M, axis=1),
                          boxes_b_x1y1[np.newaxis, :, :].repeat(N, axis=0))
    max_x2y2 = np.minimum(boxes_a_x2y2[:, np.newaxis, :].repeat(M, axis=1),
                          boxes_b_x2y2[np.newaxis, :, :].repeat(N, axis=0))

    intersection = np.clip(max_x2y2-max_x1y1, a_min=0, a_max=np.inf)
    intersection_area = intersection[:, :, 0]*intersection[:, :, 1]

    boxes_a_area = (boxes_a[:, 2]*boxes_a[:, 3])[:, np.newaxis].repeat(M, axis=1)
    boxes_b_area = (boxes_b[:, 2]*boxes_b[:, 3])[np.newaxis, :].repeat(N, axis=0)

    return intersection_area/(boxes_a_area+boxes_b_area-intersection_area)


def IoUBEV_dis_mat(boxes_a, boxes_b, gpu=True):
    """
    Args:
        boxes_a: (N*7) x, y, z, l, w, h, yaw
        boxes_b: (M*7) x, y, z, l, w, h, yaw
    Returns:
        iou matrix: (N*M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    if N == 0 or M == 0:
        iou_matrix = np.zeros((N, M), dtype=np.float32)
    else:
        iou_matrix = torch.zeros((boxes_a.shape[0], boxes_b.shape[0])).float().cuda()
        iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), iou_matrix)
        iou_matrix = iou_matrix.cpu().numpy()

    return iou_matrix


def IoU3D_dis_mat(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N*6) x, y, z, l, w, h
        boxes_b: (M*6) x, y, z, l, w, h
    Returns:
        iou matrix: (N*M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    if N == 0 or M == 0:
        iou_matrix = np.zeros((N, M), dtype=np.float32)
    else:
        iou_matrix = boxes_iou3d_gpu(boxes_a, boxes_b)
        iou_matrix = iou_matrix.cpu().numpy()

    return iou_matrix


def GIoU3D_dis_mat(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N*6) x, y, z, l, w, h
        boxes_b: (M*6) x, y, z, l, w, h
    Returns:
        iou matrix: (N*M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    N = boxes_a.shape[0]
    M = boxes_b.shape[0]

    if N == 0 or M == 0:
        iou_matrix = np.zeros((N, M), dtype=np.float32)
    else:
        iou_matrix = boxes_giou3d_gpu(boxes_a, boxes_b)
        iou_matrix = iou_matrix.cpu().numpy()

    return iou_matrix

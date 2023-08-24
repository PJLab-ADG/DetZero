from .distance import GNN_assignment, IoU2D_dis_mat, IoU3D_dis_mat, \
    IoUBEV_dis_mat, GIoU3D_dis_mat, bev_overlap_gpu

AssigenmentFunc = {
    'GNN': GNN_assignment,
}

DistanceFunc = {
    'IoU2D': IoU2D_dis_mat,
    'IoU3D': IoU3D_dis_mat,
    'IoUBEV': IoUBEV_dis_mat,
    'GIoU3D_dis_mat': GIoU3D_dis_mat,
}

__all__ = {
    'Assigenment': AssigenmentFunc,
    'Distance': DistanceFunc,
}

from .data_association import associate_det_to_tracks
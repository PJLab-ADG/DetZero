import numpy as np
import torch


class TargetAssigner(object):
    def __init__(self, anchor_sizes=None, mode='size', **kwargs):
        super().__init__()

        self.anchor_sizes = anchor_sizes
        self.anchor_slen = len(anchor_sizes) if anchor_sizes is not None else 0
        self.mode = mode
        self.dir_bin_num = 12
        self.anchor_angles = torch.arange(self.dir_bin_num,
            dtype=torch.float) * (2*np.pi/self.dir_bin_num) - np.pi 
        self.score_thresh = kwargs.get('score_thresh', [0.25, 0.5])

    def encode_torch(self, data_dict):
        target_dict = {}
        if self.mode =='geometry':
            gt_box = data_dict['gt_box']
            bs = gt_box.size(0)
            device = gt_box.device
            
            anchor_sizes = self.anchor_sizes.unsqueeze(0).repeat(bs, 1, 1).to(device)
            gt_sizes = gt_box[:, 3:6].unsqueeze(1).repeat(1, self.anchor_slen, 1)
            delta_sizes = (gt_sizes - anchor_sizes) / anchor_sizes

            target_dict["geometry_reg"] = delta_sizes.reshape(bs, -1)
            target_dict["geometry_cls"] = torch.min(torch.sum(
                torch.abs(delta_sizes), dim=-1), dim=-1)[1]

        if self.mode == 'position':
            traj = data_dict['pos_trajectory'].clone()
            traj_gt = data_dict['gt_pos_trajectory'].clone()
            bs, box_num, _ = traj_gt.size()
            device = traj.device
            
            outlier_idx = torch.where(traj_gt[:, :, 6] < -torch.pi)
            traj_gt[:, :, 6][outlier_idx] += 2*torch.pi
            outlier_idx = torch.where(traj_gt[:, :, 6] > torch.pi)
            traj_gt[:, :, 6][outlier_idx] -= 2*torch.pi

            target_dict['center_reg'] = traj_gt[:, :, :3] - traj[:, :, :3]
            
            anchor_angles = self.anchor_angles.unsqueeze(0).unsqueeze(1).repeat(bs, box_num, 1).to(device)
            gt_angles = traj_gt[:, :, 6].unsqueeze(-1).repeat(1, 1, 12)
            
            target_dict['heading_reg'] = (gt_angles - anchor_angles) / (np.pi/self.dir_bin_num)
            target_dict['heading_cls'] = torch.floor(
                (traj_gt[:, :, 6] + np.pi) / (np.pi/6.)).type(torch.LongTensor).to(device)
            
            target_dict['boxes_gt'] = traj_gt
            target_dict['box_num'] = data_dict['box_num']
        
        if self.mode == 'confidence':
            iou = torch.clamp(data_dict['iou'], 0, 1).reshape(-1)
            gt_label = torch.zeros_like(iou)
            mask = torch.zeros_like(iou)

            neg_idx = iou < self.score_thresh[0]
            mask[neg_idx] = 1
            
            pos_idx = iou >= self.score_thresh[1]
            mask[pos_idx] = 1
            gt_label[pos_idx] = 1

            target_dict['score_gt'] = gt_label
            target_dict['iou_gt'] = iou
            target_dict['mask'] = mask.type(torch.bool)

        return target_dict

    def decode_torch(self, preds_dict, data_dict):
        if self.mode == 'geometry':
            bs = preds_dict["geometry_reg"].size(0)
            device = preds_dict["geometry_reg"].device
            geo_reg = preds_dict["geometry_reg"].reshape(bs, self.anchor_slen, 3)
            
            anchor_sizes = self.anchor_sizes.unsqueeze(0).repeat(bs, 1, 1).to(device)
            geo_reg = geo_reg * anchor_sizes + anchor_sizes

            geo_cls = torch.max(preds_dict["geometry_cls"], dim=-1)[1].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3)
            geo_reg = torch.gather(geo_reg, 1, geo_cls).squeeze(1)
            
            cent_reg = torch.zeros_like(geo_reg)
            dir_reg = torch.zeros_like(geo_reg[:, 0:1])
            box_preds = torch.cat([cent_reg, geo_reg, dir_reg], dim=-1)
        
        elif self.mode in ['position', 'confidence']:
            bs, box_num, _ = preds_dict["center_reg"].shape
            device = preds_dict["center_reg"].device
            
            size_reg = preds_dict["size_reg"]   # fake size reg
            center_reg = preds_dict["center_reg"] + data_dict["pos_trajectory"][:, :, :3]

            anchor_angles = self.anchor_angles.unsqueeze(0).unsqueeze(0).repeat(bs, box_num, 1).to(device)   # B, 200, 12
            dir_reg = preds_dict["heading_reg"] * (np.pi/self.dir_bin_num) + anchor_angles  # B, 200, 12

            dir_cls = torch.max(preds_dict["heading_cls"], dim=-1)[1].unsqueeze(-1)     # B, 200, 1
            dir_reg = torch.gather(dir_reg, 2, dir_cls)
            
            box_preds = torch.cat([center_reg, size_reg, dir_reg], dim=-1) 

        return box_preds


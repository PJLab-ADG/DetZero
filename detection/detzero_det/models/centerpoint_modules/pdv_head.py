import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detzero_utils import common_utils
from detzero_utils.ops.pointnet2.pointnet2_stack.pointnet2_modules import StackSAModuleMSG, StackSAModuleMSGAttention

from detzero_det.utils import loss_utils, box_coder_utils, voxel_aggregation_utils, density_utils
from detzero_det.utils.model_nms_utils import class_agnostic_nms
from detzero_det.utils.attention_utils import TransformerEncoder, get_positional_encoder
from .proposal_target_layer import ProposalTargetLayer


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:
        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)
        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)
        Returns:
        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        if cls_preds is None:
            batch_cls_preds = None
        else:
            batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds


class VoxelAggregationHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg = model_cfg.ROI_GRID_POOL
        layer_cfg = self.pool_cfg.POOL_LAYERS
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        c_out = 0
        self.roi_grid_pool_layers = nn.ModuleList()
        for i, src_name in enumerate(self.pool_cfg.FEATURE_LOCATIONS):
            mlps = layer_cfg[src_name].MLPS
            for k in range(len(mlps)):
                mlps[k] = [self.model_cfg.VOXEL_AGGREGATION.NUM_FEATURES[i]] + mlps[k]
            stack_sa_module_msg = StackSAModuleMSGAttention if self.pool_cfg.get('ATTENTION', {}).get('ENABLED') else StackSAModuleMSG
            pool_layer = stack_sa_module_msg(
                radii=layer_cfg[src_name].POOL_RADIUS,
                nsamples=layer_cfg[src_name].NSAMPLE,
                mlps=mlps,
                use_xyz=True,
                pool_method=layer_cfg[src_name].POOL_METHOD,
                use_density=layer_cfg[src_name].get('USE_DENSITY')
            )

            self.roi_grid_pool_layers.append(pool_layer)
            c_out += sum([x[-1] for x in mlps])

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            assert self.pool_cfg.ATTENTION.NUM_FEATURES == c_out, f'ATTENTION.NUM_FEATURES must equal voxel aggregation output dimension of {c_out}.'
            pos_encoder = get_positional_encoder(self.pool_cfg)
            self.attention_head = TransformerEncoder(self.pool_cfg.ATTENTION, pos_encoder)

            # TODO: Check if this is necessary
            # Hack for xavier initialization (https://github.com/pashu123/Transformers/blob/master/train.py#L26-L29)
            for p in self.attention_head.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )

        # Density confidence predction
        if self.model_cfg.get('DENSITY_CONFIDENCE', {}).get('ENABLED'):
            self.cls_layers = self.make_fc_layers(
                input_channels=(3 +
                                self.model_cfg.DENSITY_CONFIDENCE.GRID_SIZE ** 3 +
                                (pre_channel if self.model_cfg.DENSITY_CONFIDENCE.ADD_SHARED_FEATURES else 0)),
                output_channels = self.num_class,
                fc_list=self.model_cfg.CLS_FC
            )
        else:
            self.cls_layers = self.make_fc_layers(
                input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
            )

        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        batch_rois = batch_dict['rois']

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            batch_dict, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)

        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)
        new_xyz = global_roi_grid_points.view(-1, 3)

        pooled_features_list = []
        ball_idxs_list = []
        for k, src_name in enumerate(self.pool_cfg.FEATURE_LOCATIONS):
            point_coords = batch_dict['point_coords'][src_name]
            point_features = batch_dict['point_features'][src_name]
            pool_layer = self.roi_grid_pool_layers[k]

            xyz = point_coords[:, 1:4]
            xyz_batch_cnt = xyz.new_zeros(batch_size).int()
            batch_idx = point_coords[:, 0]
            for k in range(batch_size):
                xyz_batch_cnt[k] = (batch_idx == k).sum()

            new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
            pool_output = pool_layer(
                xyz=xyz.contiguous(),
                xyz_batch_cnt=xyz_batch_cnt,
                new_xyz=new_xyz,
                new_xyz_batch_cnt=new_xyz_batch_cnt,
                features=point_features.contiguous(),
            )  # (M1 + M2 ..., C)

            if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
                _, pooled_features, ball_idxs = pool_output
            else:
                _, pooled_features = pool_output

            pooled_features = pooled_features.view(
                -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
                pooled_features.shape[-1]
            )  # (BxN, 6x6x6, C)
            pooled_features_list.append(pooled_features)

            if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
                ball_idxs = ball_idxs.view(
                    -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE **3,
                    ball_idxs.shape[-1]
                )
                ball_idxs_list.append(ball_idxs)

        all_pooled_features = torch.cat(pooled_features_list, dim=-1)
        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            all_ball_idxs = torch.cat(ball_idxs_list, dim=-1)
        else:
            all_ball_idxs = []
        return all_pooled_features, global_roi_grid_points, local_roi_grid_points, all_ball_idxs

    def get_global_grid_points_of_roi(self, batch_dict, grid_size):
        rois = batch_dict['rois']
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)

        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def get_point_voxel_features(self, batch_dict):
        raise NotImplementedError

    def get_positional_input(self, points, rois, local_roi_grid_points):
        points_per_part = density_utils.find_num_points_per_part_multi(points,
                                                                       rois,
                                                                       self.model_cfg.ROI_GRID_POOL.GRID_SIZE,
                                                                       self.pool_cfg.ATTENTION.MAX_NUM_BOXES,
                                                                       return_centroid=self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_centroid')
        points_per_part_num_features = 1 if len(points_per_part.shape) <= 5 else points_per_part.shape[-1]
        points_per_part = points_per_part.view(points_per_part.shape[0]*points_per_part.shape[1], -1, points_per_part_num_features).float()
        # First feature is density, other potential features are xyz
        points_per_part[..., 0] = torch.log10(points_per_part[..., 0] + 0.5) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
        if self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'grid_points':
            positional_input = local_roi_grid_points
        elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density':
            positional_input = points_per_part
        elif self.pool_cfg.ATTENTION.POSITIONAL_ENCODER == 'density_grid_points':
            positional_input = torch.cat((local_roi_grid_points, points_per_part), dim=-1)
        else:
            positional_input = None
        return positional_input

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """
        batch_dict['point_features'], batch_dict['point_coords'] = self.get_point_voxel_features(batch_dict)

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features, global_roi_grid_points, local_roi_grid_points, ball_idxs = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)
        batch_size_rcnn = pooled_features.shape[0]

        if self.pool_cfg.get('ATTENTION', {}).get('ENABLED'):
            src_key_padding_mask = None
            if self.pool_cfg.ATTENTION.get('MASK_EMPTY_POINTS'):
                src_key_padding_mask = (ball_idxs == 0).all(-1)

            positional_input = self.get_positional_input(batch_dict['points'], batch_dict['rois'], local_roi_grid_points)
            # Attention
            attention_output = self.attention_head(pooled_features, positional_input, src_key_padding_mask) # (BxN, 6x6x6, C)

            if self.pool_cfg.ATTENTION.get('COMBINE'):
                attention_output = pooled_features + attention_output

            # Permute
            grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
            batch_size_rcnn = attention_output.shape[0]
            pooled_features = attention_output.permute(0, 2, 1).\
                contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size) # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if self.model_cfg.get('DENSITY_CONFIDENCE', {}).get('ENABLED'):
            with torch.no_grad():
                # Calculate number of points in each rcnn_reg
                _, batch_box_preds = self.generate_predicted_boxes(
                    batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=None, box_preds=rcnn_reg
                )
                points_per_part = density_utils.find_num_points_per_part_multi(batch_dict['points'],
                                                                               batch_box_preds,
                                                                               self.model_cfg.DENSITY_CONFIDENCE.GRID_SIZE,
                                                                               self.model_cfg.DENSITY_CONFIDENCE.MAX_NUM_BOXES)
                points_per_part = torch.log10(points_per_part.float() + 0.5).reshape(-1, self.model_cfg.DENSITY_CONFIDENCE.GRID_SIZE ** 3, 1) - (math.log10(0.5) if self.model_cfg.get('DENSITY_LOG_SHIFT') else 0)
                point_cloud_range = torch.tensor(self.point_cloud_range, device=batch_box_preds.device)
                batch_box_preds_xyz = batch_box_preds.reshape(-1, batch_box_preds.shape[-1], 1)[:, :3]
                batch_box_preds_xyz /= (point_cloud_range[3:] - point_cloud_range[:3]).unsqueeze(0).unsqueeze(-1)

                density_features = [points_per_part, batch_box_preds_xyz]
                if self.model_cfg.DENSITY_CONFIDENCE.ADD_SHARED_FEATURES:
                    density_features.append(shared_features)

            density_features = torch.cat(density_features, dim=1)
            rcnn_cls = self.cls_layers(density_features)  # (B, 1 or 2)
        else:
            rcnn_cls = self.cls_layers(shared_features)

        rcnn_cls = rcnn_cls.transpose(1, 2).contiguous().squeeze(dim=1)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict


class VoxelCenterHead(VoxelAggregationHead):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, point_cloud_range, voxel_size, num_class, kwargs=kwargs)

    def get_point_voxel_features(self, batch_dict):
        point_features = {}
        point_coords = {}
        for feature_location in self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS:
            # Voxel aggregation based on voxel centers
            cur_coords = batch_dict['multi_scale_3d_features'][feature_location].indices
            xyz = common_utils.get_voxel_centers(
                cur_coords[:, 1:4],
                downsample_times=batch_dict['multi_scale_3d_strides'][feature_location],
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
            cur_coords = cur_coords.type(torch.cuda.FloatTensor)
            cur_coords[:, 1:4] = xyz

            # Input features for grid pooling module
            point_features[feature_location] = batch_dict['multi_scale_3d_features'][feature_location].features
            point_coords[feature_location] = cur_coords
        return point_features, point_coords


class PDVHead(VoxelAggregationHead):
    def __init__(self, input_channels, model_cfg, point_cloud_range, voxel_size, num_class=1, **kwargs):
        super().__init__(input_channels, model_cfg, point_cloud_range, voxel_size, num_class, kwargs=kwargs)

    def get_point_voxel_features(self, batch_dict):
        point_features = {}
        point_coords = {}
        centroids_all, centroid_voxel_idxs_all = voxel_aggregation_utils.get_centroids_per_voxel_layer(batch_dict['points'],
                                                                                                       self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS,
                                                                                                       batch_dict['multi_scale_3d_strides'],
                                                                                                       self.voxel_size,
                                                                                                       self.point_cloud_range)
        for feature_location in self.model_cfg.VOXEL_AGGREGATION.FEATURE_LOCATIONS:
            centroids = centroids_all[feature_location][:, :4]
            centroid_voxel_idxs = centroid_voxel_idxs_all[feature_location]
            x_conv = batch_dict['multi_scale_3d_features'][feature_location]
            overlapping_voxel_feature_indices_nonempty, overlapping_voxel_feature_nonempty_mask = \
                voxel_aggregation_utils.get_nonempty_voxel_feature_indices(centroid_voxel_idxs, x_conv)

            if self.model_cfg.VOXEL_AGGREGATION.get('USE_EMPTY_VOXELS'):
                voxel_points = torch.zeros((x_conv.features.shape[0], centroids.shape[-1]), dtype=centroids.dtype, device=centroids.device)
                voxel_points[overlapping_voxel_feature_indices_nonempty] = centroids[overlapping_voxel_feature_nonempty_mask]

                # Set voxel center
                empty_mask = torch.ones((x_conv.features.shape[0]), dtype=torch.bool, device=centroids.device)
                empty_mask[overlapping_voxel_feature_indices_nonempty] = False
                cur_coords = x_conv.indices[empty_mask]
                xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=batch_dict['multi_scale_3d_strides'][feature_location],
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )
                cur_coords = cur_coords.type(torch.cuda.FloatTensor)
                cur_coords[:, 1:4] = xyz
                voxel_points[empty_mask] = cur_coords

                point_features[feature_location] = x_conv.features
                point_coords[feature_location] = voxel_points
            else:
                x_conv_features = torch.zeros((centroids.shape[0], x_conv.features.shape[-1]), dtype=x_conv.features.dtype, device=centroids.device)
                x_conv_features[overlapping_voxel_feature_nonempty_mask] = x_conv.features[overlapping_voxel_feature_indices_nonempty]

                point_features[feature_location] = x_conv_features[overlapping_voxel_feature_nonempty_mask]
                point_coords[feature_location] = centroids[overlapping_voxel_feature_nonempty_mask]
        return point_features, point_coords

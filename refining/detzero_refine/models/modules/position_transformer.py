import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from detzero_utils.model_utils import make_linear_layers, make_fc_layers, make_conv_layers
from detzero_utils import common_utils

from detzero_refine.utils.mmdet_utils import FocalLoss
from detzero_refine.models.modules.head import PositionHead
from detzero_refine.models.modules.target_assign import TargetAssigner


class PositionTransformer(nn.Module):
    def __init__(self, model_cfg, query_point_dims=None, memory_point_dims=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.query_point_dims = query_point_dims
        self.memory_point_dims = memory_point_dims
        self.embed_dims = self.model_cfg.get('EMBED_DIMS', 256)
        
        self.target_assigner = TargetAssigner(mode='position')

        loss_cls = model_cfg.LOSS_CLS
        if loss_cls['type'] == 'FocalLoss':
            loss_cls.pop('type')
            self.loss_cls = FocalLoss(**loss_cls)
        elif loss_cls['type'] == 'CrossEntropyLoss':
            loss_cls.pop('type')
            self.loss_cls = nn.CrossEntropyLoss(**loss_cls)
        else:
            raise NotImplementedError('Not implemented Classification Loss.')
        
        self.loss_bbox = nn.L1Loss(reduction='none')
        self.loss_weight = [0.1, 2]

        self.preds_dict = {}
        self.targets_dict = {}

        self._init_layers()
        self.init_attn_mask(half_windows=3)

    def _init_layers(self):
        # query encoder
        self.query_encoder = make_conv_layers(
            self.model_cfg.QUERY_ENCODER,
            self.query_point_dims,
            self.embed_dims,
            output_use_norm=True
        )

        self.query_mlp = make_fc_layers(
            self.model_cfg.REGRESSION_MLP,
            self.embed_dims,
            self.embed_dims,
            output_use_norm=True
        )

        # memory encoder
        self.memory_encoder = make_fc_layers(
            self.model_cfg.MEMORY_ENCODER,
            self.memory_point_dims,
            self.embed_dims,
            output_use_norm=True
        )

        self._points_intermediate = torch.empty(0)
        self.memory_encoder[5].register_forward_hook(self.save_points_intermeidate())

        self.memory_mlp = make_fc_layers(
            self.model_cfg.REGRESSION_MLP,
            self.embed_dims + self.model_cfg.MEMORY_ENCODER[1],
            self.embed_dims,
            output_use_norm=True
        )

        # decoder
        self.decoder_cfg = self.model_cfg.get('DECODER', None)
        if self.decoder_cfg['NAME'] == 'PositionHead':
            self.decoder = PositionHead(**self.decoder_cfg)

    def init_attn_mask(self, half_windows=3, box_num=200, pts_num=256):
        self.attn_mask = torch.zeros((box_num + 2 * half_windows) * box_num)
        windows = 2 * half_windows + 1
        for i in range(windows):
            self.attn_mask[i::(box_num + 2 * half_windows+1)] = 1
        self.attn_mask = self.attn_mask.reshape(box_num, box_num + 2 * half_windows)
        self.attn_mask = self.attn_mask[:, half_windows:-half_windows]
        self.attn_mask = self.attn_mask.unsqueeze(-1)
        self.attn_mask = self.attn_mask.repeat(1, 1, pts_num)
        self.attn_mask = self.attn_mask.reshape(box_num, -1)

    def save_points_intermeidate(self):
        def fn(_, __, output):
            self._points_intermediate = output
        return fn
    
    def forward(self, data_dict):

        local_pts = data_dict['pos_query_points']
        global_pts = data_dict['pos_memory_points']
        traj = data_dict['pos_trajectory']
        bs, box_num, pts_num, _ = local_pts.size()
        device = local_pts.device
        
        # prepare query features
        local_pts = local_pts.permute(0, 3, 1, 2)
        q_feat = self.query_encoder(local_pts)
        q_feat = torch.max(q_feat, dim=3, keepdim=False)[0]
        q_feat = self.query_mlp(q_feat)
        # prepare query pos
        query_pos = torch.cat([traj[..., :3], traj[..., 6:]], dim=-1)

        # prepare memory features
        global_pts_num = global_pts.size(2)
        global_pts = global_pts.permute(0, 3, 1, 2).reshape(bs, -1, box_num*global_pts_num)
        m_feat = self.memory_encoder(global_pts)
        m_feat = torch.max(m_feat, dim=2, keepdim=True)[0]
        m_feat = torch.cat([m_feat.repeat(1, 1, global_pts.size(2)), self._points_intermediate], dim=1)
        m_feat = self.memory_mlp(m_feat)

        data_dict['query'] = q_feat     # [bs, 256, 200]
        data_dict['memory'] = m_feat      # [bs, 256, 200*128]
        data_dict['query_pos'] = query_pos

        preds_dict = self.decoder(data_dict)

        for key in preds_dict.keys():
            preds_dict[key] = preds_dict[key].permute(0, 2, 1)
        preds_dict['size_reg'] = data_dict['pos_trajectory'][:, :, 3:6]
        self.preds_dict.update(preds_dict)

        if self.training:
            targets_dict = self.target_assigner.encode_torch(data_dict)
            self.targets_dict.update(targets_dict)
        else:
            batch_box_preds = self.target_assigner.decode_torch(preds_dict, data_dict)
            data_dict['batch_box_preds'] = batch_box_preds
            preds_dict['batch_box_preds'] = batch_box_preds

        return data_dict

    def get_loss(self, tb_dict=None):
        bs = self.preds_dict['center_reg'].size(0)
        box_num = self.targets_dict['box_num']
        reg_loss = self.loss_bbox
        cls_loss = self.loss_cls

        # center loss
        cent_reg_loss = 0
        for i in range(len(box_num)):
            num = box_num[i]
            cent_reg_loss += torch.sum(reg_loss(
                self.preds_dict['center_reg'][i, :num, :],
                self.targets_dict['center_reg'][i, :num, :]
            )) / bs / num

        # direction loss
        dir_cls_loss = 0
        dir_reg_loss = 0 
        for i in range(len(box_num)):
            num = box_num[i]
            dir_cls_loss += cls_loss(
                self.preds_dict['heading_cls'][i, :num, :],
                self.targets_dict['heading_cls'][i, :num]
            ) / bs

            dir_tmp_loss = reg_loss(
                self.preds_dict['heading_reg'][i, :num, :],
                self.targets_dict['heading_reg'][i, :num, :]
            )
            dir_tmp_loss = torch.gather(dir_tmp_loss, 1,
                self.targets_dict["heading_cls"][i, :num].unsqueeze(-1))
            dir_reg_loss += dir_tmp_loss.sum() / bs / num

        tb_dict.update({
            'center_reg_loss': cent_reg_loss,
            'heading_cls_loss': dir_cls_loss,
            'heading_reg_loss': dir_reg_loss
        })

        w1, w2 = self.loss_weight
        pos_loss = cent_reg_loss + w1*dir_cls_loss + w2*dir_reg_loss
        tb_dict['position_loss'] = pos_loss

        return pos_loss, tb_dict
                                 
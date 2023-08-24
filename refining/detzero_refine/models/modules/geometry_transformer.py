import torch
import torch.nn as nn
import torch.nn.functional as F

from detzero_utils.model_utils import make_linear_layers, make_fc_layers

from detzero_refine.models.modules.head import GeometryHead
from detzero_refine.models.modules.target_assign import TargetAssigner


class GeometryTransformer(nn.Module):
    def __init__(self, model_cfg, query_point_dims=None, memory_point_dims=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.query_point_dims = query_point_dims
        self.memory_point_dims = memory_point_dims
        self.embed_dims = self.model_cfg.get('EMBED_DIMS', 256)
        self.anchor_sizes = self.model_cfg.get('ANCHOR_SIZES',
            [[4.8, 1.8, 1.5], [10.0, 2.6, 3.2], [2.0, 1.0, 1.6]])

        self.target_assigner = TargetAssigner(
            anchor_sizes=torch.tensor(self.anchor_sizes, dtype=torch.float),
            mode='geometry'
        )
        self.preds_dict = {}
        self.targets_dict = {}

        self.loss_weight = [0.1, 2]
        self.reg_loss = nn.L1Loss(reduction='none')
        self.cls_loss = nn.CrossEntropyLoss(reduction='mean')

        self._init_layers()

    def _init_layers(self):
        # memory encoder
        self.memory_encoder = make_fc_layers(
            self.model_cfg.MEMORY_ENCODER,
            self.query_point_dims,
            self.embed_dims*2,
            output_use_norm=True
        )
        self.memory_mlp = make_fc_layers(
            self.model_cfg.REGRESSION_MLP,
            self.embed_dims*2 + self.model_cfg.MEMORY_ENCODER[1],
            self.embed_dims,
            output_use_norm=True
        )
        self._points_intermediate = torch.empty(0)
        self.memory_encoder[5].register_forward_hook(self.save_points_intermeidate())

        # query encoder
        self.query_encoder = make_fc_layers(
            self.model_cfg.QUERY_ENCODER,
            self.memory_point_dims,
            self.embed_dims,
            output_use_norm=True
        )
        self.query_mlp = make_linear_layers(
            self.model_cfg.REGRESSION_MLP,
            self.embed_dims,
            self.embed_dims,
            output_use_norm=True
        )

        # decoder
        self.decoder_cfg = self.model_cfg.get('DECODER', None)
        if self.decoder_cfg['NAME'] == 'GeometryHead':
            self.decoder = GeometryHead(**self.decoder_cfg)

    def save_points_intermeidate(self):
        def fn(_, __, output):
            self._points_intermediate = output
        return fn

    def assign_targets(self, data_dict):
        gt_box = data_dict['gt_geo_query_boxes']
        query_num = gt_box.size(1)
        geo_cls, geo_reg = [], []
        for i in range(query_num):
            temp_dict = {'gt_box': gt_box[:, i]}
            res_dict = self.target_assigner.encode_torch(temp_dict)
            geo_cls.append(res_dict['geometry_cls'])
            geo_reg.append(res_dict['geometry_reg'])

        targets_dict = {
            'geometry_cls': torch.stack(geo_cls, dim=1),
            'geometry_reg': torch.stack(geo_reg, dim=1)
        }
        return targets_dict

    def generate_predicted_boxes(self, preds_dict, data_dict):
        layer_num, bs, query_num, _ = preds_dict['geometry_cls'].size()
        query_num_ori = preds_dict['geo_query_num']
        layer_boxes = []
        for idx_layer in range(layer_num):
            geo_cls = preds_dict['geometry_cls'][idx_layer]
            geo_reg = preds_dict['geometry_reg'][idx_layer]
            query_boxes = []
            for i in range(query_num):
                temp_dict = {
                    'geometry_cls': geo_cls[:, i],
                    'geometry_reg': geo_reg[:, i]
                }
                boxes = self.target_assigner.decode_torch(temp_dict, data_dict)
                query_boxes.append(boxes)

            query_boxes = torch.stack(query_boxes, dim=1)
            batch_boxes = []
            for idx in range(bs):
                batch_boxes.append(query_boxes[idx, :query_num_ori[idx]].mean(dim=0))

            layer_boxes.append(torch.stack(batch_boxes, dim=0))
        
        layer_boxes = torch.stack(layer_boxes, dim=0)
        res = layer_boxes.mean(0)
        return res
    
    def forward(self, data_dict):
        m_pts = data_dict['geo_memory_points'].permute(0, 2, 1)
        bs, channel, pts_num = m_pts.size()

        # generate memory features for cross_attn
        m_feat = self.memory_encoder(m_pts)
        m_feat = torch.max(m_feat, dim=2, keepdim=True)[0].repeat(1, 1, pts_num)
        m_feat = torch.cat([self._points_intermediate, m_feat], dim=1)
        m_feat = self.memory_mlp(m_feat)

        # generate local point features of init_box as query for self_attn
        q_pts = data_dict['geo_query_points'].clone()
        if q_pts.dim() == 4:
            bs, pp_num, pts_num, dim = q_pts.size()
            q_pts = q_pts.reshape(bs*pp_num, pts_num, dim)
            q_pts = q_pts.permute(0, 2, 1)
        
        q_feat = self.query_encoder(q_pts)
        q_feat = torch.max(q_feat, dim=2, keepdim=False)[0]
        q_feat = self.query_mlp(q_feat)

        data_dict['query'] = q_feat.reshape(bs, pp_num, -1).permute(0, 2, 1)
        data_dict['memory'] = m_feat
        
        preds_dict = self.decoder(data_dict)[0][0]
        for key in preds_dict.keys():
            preds_dict[key] = preds_dict[key].permute(0, 1, 3, 2)
        
        preds_dict['geo_query_num'] = data_dict['geo_query_num']
        self.preds_dict.update(preds_dict)

        if self.training:
            targets_dict = self.assign_targets(data_dict)
            self.targets_dict.update(targets_dict)
        else:
            batch_box_preds = self.generate_predicted_boxes(preds_dict, data_dict)
            data_dict['batch_box_preds'] = batch_box_preds

        return data_dict
    
    def get_loss(self, tb_dict=None):
        layer_num, bs, max_num, _ = self.preds_dict["geometry_reg"].size()
        
        cls_loss, reg_loss = 0, 0
        for idx_layer in range(layer_num):
            for i in range(bs):
                query_num = self.preds_dict['geo_query_num'][i]

                cls_loss += self.cls_loss(
                    self.preds_dict["geometry_cls"][idx_layer, i, :query_num],
                    self.targets_dict["geometry_cls"][i, :query_num]
                )

                reg_loss_temp = self.reg_loss(
                    self.preds_dict["geometry_reg"][idx_layer, i, :query_num],
                    self.targets_dict["geometry_reg"][i, :query_num]
                ).reshape(query_num, -1, 3)

                reg_loss_temp = torch.gather(reg_loss_temp, 1,
                    self.targets_dict["geometry_cls"][i, :query_num].unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 3))

                reg_loss_temp = reg_loss_temp.sum() / query_num
                reg_loss += reg_loss_temp

        cls_loss /= bs
        reg_loss /= bs

        tb_dict.update({
            "reg_loss": reg_loss,
            "cls_loss": cls_loss,
        })

        w1, w2 = self.loss_weight
        loss = w1 * cls_loss + w2 * reg_loss
        tb_dict["geometry_loss"] = loss
        
        return loss, tb_dict

import torch
import torch.nn as nn

from detzero_utils.model_utils import make_fc_layers, make_conv_layers

from detzero_refine.models.modules.target_assign import TargetAssigner


class ConfidencePointnet(nn.Module):
    def __init__(self, model_cfg, query_point_dims=None, memory_point_dims=None):
        super().__init__()
        self.model_cfg = model_cfg
        self.embed_dims = self.model_cfg.get('EMBED_DIMS', 256)

        self.pts_encoder_1 = make_conv_layers(
            self.model_cfg.ENCODER_MLP,
            query_point_dims,
            self.embed_dims,
            output_use_norm=True
        )
        self.pts_encoder_2 = make_conv_layers(
            [],
            self.embed_dims+self.model_cfg.ENCODER_MLP[1],
            self.embed_dims,
            output_use_norm=True
        )
        self._pts_inter = torch.empty(0)
        self.pts_encoder_1[5].register_forward_hook(self.save_points_intermeidate())

        self.pts_mlp = make_fc_layers(
            self.model_cfg.REGRESSION_MLP,
            self.embed_dims,
            self.embed_dims,
            output_use_norm=True
        )
        self.regression_mlp = make_fc_layers(
            self.model_cfg.REGRESSION_MLP,
            self.embed_dims*2,
            self.embed_dims,
            output_use_norm=True
        )
        self._pool_inter = torch.empty(0)
        self.pts_mlp[5].register_forward_hook(self.save_joint_intermeidate())

        self.heads = nn.ModuleDict()
        self.preds_dict = {}
        self.targets_dict = {}

        self.score_thresh = model_cfg.get('SCORE_THRESH', [0.25, 0.5])
        self.bce_loss = nn.BCELoss(reduction='none')
        self.loss_weight = [1.0, 1.0]

        self.target_assigner = TargetAssigner(
            mode='confidence',
            score_thresh=self.score_thresh
        )

        self.tasks = {
            "score_reg": 1,
            "iou_reg": 1
        }
        for task in self.tasks:
            self.heads[task] = make_fc_layers(
                [int(self.embed_dims/2)],
                self.embed_dims,
                self.tasks[task],
                output_use_norm=False
            )

    def save_points_intermeidate(self):
        def fn(_, __, output):
            self._pts_inter = output
        return fn

    def save_joint_intermeidate(self):
        def fn(_, __, output):
            self._pool_inter = output
        return fn

    def forward(self, data_dict):
        points = data_dict['conf_points']
        bs, box_num, pts_num, _ = points.size()                     # B, 200, 256, 32
        points = points.permute(0, 3, 1, 2)                         # B, 32, 200, 256
        
        pts_feat = self.pts_encoder_1(points)                       # B, 256, 200, 256
        pts_feat = torch.max(pts_feat, dim=3, keepdim=False)[0]     # B, 256, 200
        pts_feat = pts_feat.unsqueeze(-1).repeat(1, 1, 1, pts_num)  # B, 256, 200, 256
        
        pts_feat = torch.cat([pts_feat, self._pts_inter], dim=1)    # B, 384, 200, 256
        pts_feat = self.pts_encoder_2(pts_feat)                     # B, 256, 200, 256
        
        pool_feat = torch.max(pts_feat, dim=3, keepdim=False)[0]    # B, 256, 200
        pool_feat = self.pts_mlp(pool_feat)                         # B, 256, 200

        pool_feat = torch.max(pool_feat, dim=2, keepdim=False)[0]   # B, 256
        pool_feat = pool_feat.unsqueeze(-1).repeat(1, 1, box_num)   # B, 256, 200
        pool_feat = torch.cat([pool_feat, self._pool_inter], dim=1) # B, 512, 200
        output = self.regression_mlp(pool_feat)                     # B, 256, 200

        preds_dict = {}
        for task in self.tasks:
            res = self.heads[task](output).permute(0, 2, 1)
            preds_dict[task] = torch.sigmoid(res)
        self.preds_dict.update(preds_dict)

        if self.training:
            targets_dict = self.target_assigner.encode_torch(data_dict)
            self.targets_dict.update(targets_dict)
        else:
            data_dict['pred_score'] = torch.sqrt(
                preds_dict['score_reg'].squeeze(2) * preds_dict['iou_reg'].squeeze(2))

        return data_dict
    
    def get_loss(self, tb_dict=None):
        bs, max_box_num, _ = self.preds_dict["score_reg"].size()

        mask = self.targets_dict['mask']
        gt_label = self.targets_dict['score_gt']
        gt_label = gt_label[mask]
        macthed_box_num = gt_label.size(0)

        # positive / negative classification loss
        pred_score = self.preds_dict['score_reg'].reshape(-1)
        pred_score = pred_score[mask]
        
        bin_cls_loss = self.bce_loss(pred_score, gt_label)
        bin_cls_loss = bin_cls_loss.sum() / macthed_box_num

        # iou regression loss
        gt_iou = self.targets_dict['iou_gt']
        gt_iou = gt_iou[mask]
        
        pred_iou = self.preds_dict['iou_reg'].reshape(-1)
        pred_iou = pred_iou[mask]
        
        iou_loss = self.bce_loss(pred_iou, gt_iou)
        iou_loss = iou_loss.sum() / macthed_box_num

        w1, w2 = self.loss_weight
        conf_loss = bin_cls_loss*w1 + iou_loss*w2

        tb_dict.update({
            "bin_cls_loss": bin_cls_loss,
            "iou_loss": iou_loss,
        })

        tb_dict["confidence_loss"] = conf_loss

        return conf_loss, tb_dict


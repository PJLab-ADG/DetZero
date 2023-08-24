import numpy as np
import torch
import torch.nn as nn

from detzero_utils.ops.iou3d_nms import iou3d_nms_utils

from detzero_refine.models.modules import GeometryTransformer
from detzero_refine.models.modules import PositionTransformer
from detzero_refine.models.modules import ConfidencePointnet

__all__ = {
    'GeometryTransformer': GeometryTransformer,
    'PositionTransformer': PositionTransformer,
    'ConfidencePointnet': ConfidencePointnet
}


class RefineTemplate(nn.Module):
    def __init__(self, model_cfg, dataset):
        super().__init__()
        self.model_cfg = model_cfg
        self.dataset = dataset
        self.tta = self.dataset.tta

        self.register_buffer('global_step', torch.LongTensor(1).zero_())
        self.build_networks()

    @property
    def mode(self):
        return 'TRAIN' if self.training else 'TEST'

    def update_global_step(self):
        self.global_step += 1

    def build_networks(self):
        model_info_dict = {
            'module_list': [],
            'query_point_dims': self.model_cfg.get('QUERY_POINT_DIMS', 0),
            'memory_point_dims': self.model_cfg.get('MEMORY_POINT_DIMS', 0),
        }

        name = self.model_cfg.REGRESSION['NAME']
        reg = __all__[name](
            model_cfg=self.model_cfg.REGRESSION,
            query_point_dims=model_info_dict['query_point_dims'],
            memory_point_dims=model_info_dict['memory_point_dims']
        )
        self.add_module('reg', reg)

        return model_info_dict

    def get_training_loss(self):
        disp_dict = {}
        tb_dict = {}
        loss = 0

        reg_loss, tb_dict = self.reg.get_loss(tb_dict)
        loss += reg_loss

        tb_dict["full_loss"] = loss

        return loss, tb_dict, disp_dict

    def forward(self, data_dict):
        data_dict = self.reg(data_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        
        else:
            pred_dicts, recall_dict = self.post_processing(data_dict)
            return pred_dicts, recall_dict, {}

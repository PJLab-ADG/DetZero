import copy

import torch
import torch.nn as nn

from detzero_refine.utils.mmdet_utils import multi_apply

from detzero_refine.models.modules.transformer import (PositionEmbeddingLearned,
                                                       FFN,
                                                       TransformerDecoderLayer)


class GeometryHead(nn.Module):
    def __init__(self,
                 num_classes=3,
                 num_decoder_layers=1,
                 auxiliary=True,
                 cross_only=False,
                 memory_self_attn=False,
                 num_heads=8,
                 hidden_channel=256,
                 ffn_channel=256,
                 dropout=0.1,
                 bn_momentum=0.1,
                 activation='relu',
                 bias='auto',
                 **kwargs):
        super(GeometryHead, self).__init__()

        self.num_classes = num_classes
        self.auxiliary = auxiliary
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum

        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    d_model=hidden_channel,
                    nhead=num_heads,
                    dim_feedforward=ffn_channel,
                    dropout=dropout,
                    activation=activation,
                    self_posembed=PositionEmbeddingLearned(3, hidden_channel),
                    cross_only=cross_only
                ))

        common_heads = {
            "geometry_cls": (self.num_classes, 2),
            "geometry_reg": (self.num_classes*3, 2)
        }
        conv_cfg = dict(type='Conv1d')
        norm_cfg = dict(type='BN1d')
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            self.prediction_heads.append(
                FFN(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias
                ))

        self.init_weights()
        self.forward_ret_dict = {}

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def forward_single(self, query_feat, memory, query_pos=None, memory_pos=None):
        """
        query: [B, C, num_queries]
        memory: [B, C2, num_queries*256]
        query_pos: [B, num_queries, 3]
        memory_pos: [B, num_queries*256, 3]
        """
        bs, _, N = query_feat.shape

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers-1) else f'{i}head_'

            query_feat = self.decoder[i](query_feat, memory, query_pos, memory_pos)

            # Prediction
            res_layer = self.prediction_heads[i](query_feat)
            ret_dicts.append(res_layer)

            # do not support next level positional embedding

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return [ret_dicts[-1]]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            new_res[key] = torch.stack([ret_dict[key] for ret_dict in ret_dicts])
        return [new_res]

    def forward(self, data_dict):
        query = [data_dict['query']]
        memory = [data_dict['memory']]
        query_pos = [data_dict['geo_query_boxes'][..., 3:6]]
        memory_pos = [None]

        preds_dicts = multi_apply(
            self.forward_single,
            query,
            memory,
            query_pos,
            memory_pos
        )

        return preds_dicts

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from detzero_refine.models.modules.transformer import (PositionEmbeddingLearned,
                                                       FFN,
                                                       TransformerDecoderLayer)


class PositionHead(nn.Module):
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
        super(PositionHead, self).__init__()

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
                    self_posembed=PositionEmbeddingLearned(4, hidden_channel),
                    cross_only=cross_only
                ))

        common_heads = {
            "center_reg": (3, 2),
            "heading_cls": (12, 2),
            "heading_reg": (12, 2)
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
    
    def forward(self, data_dict):
        query = data_dict['query']
        memory = data_dict['memory']
        key_padding_mask = data_dict['padding_mask'].type(torch.bool)
        query_pos = data_dict['query_pos']
        memory_pos = None
        attn_mask = None
        bs, traj_len = key_padding_mask.size()
        ca_padding_mask = key_padding_mask.reshape(bs, traj_len, 1).repeat(1, 1, 48).reshape(bs, -1)

        ret_dicts = []
        for i in range(self.num_decoder_layers):
            query = self.decoder[i].forward(
                query, memory, query_pos, memory_pos, key_padding_mask, ca_padding_mask, attn_mask)

            # Prediction
            res_layer = self.prediction_heads[i](query)
            ret_dicts.append(res_layer)

            # for next level positional embedding
            # query_pos = res_layer['center'].detach().clone().permute(0, 2, 1)

        if self.auxiliary is False:
            # only return the results of last decoder layer
            return ret_dicts[-1]

        # return all the layer's results for auxiliary superivison
        new_res = {}
        for key in ret_dicts[0].keys():
            new_res[key] = ret_dicts[0][key]

        return new_res

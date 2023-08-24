import math

import torch
import torch.nn as nn


class TransformerEncoder(nn.Module):
    """
    Based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html (https://knowyourmeme.com/memes/i-made-this)
    """
    def __init__(self, attention_cfg, pos_encoder=None):
        super().__init__()
        self.attention_cfg = attention_cfg
        self.pos_encoder = pos_encoder
        encoder_layers = nn.TransformerEncoderLayer(attention_cfg.NUM_FEATURES, attention_cfg.NUM_HEADS, attention_cfg.NUM_HIDDEN_FEATURES, attention_cfg.DROPOUT)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, attention_cfg.NUM_LAYERS)

    def forward(self, point_features, positional_input, src_key_padding_mask=None):
        """
        Args:
            point_features: (b, xyz, f)
            positional_input: (b, xyz, 3)
            src_key_padding_mask: (b, xyz)
        Returns:
            point_features: (b, xyz, f)
        """
        # Clone point features to prevent mutating input arguments
        attended_features = torch.clone(point_features)
        if src_key_padding_mask is not None:
            # RoIs sometimes have all zero inputs. This results in a 0/0 division because of masking. Thus, we
            # remove the empty rois to prevent this issue(https://github.com/pytorch/pytorch/issues/24816#issuecomment-524415617)
            empty_rois_mask = src_key_padding_mask.all(-1)
            attended_features_filtered = attended_features[~empty_rois_mask]

            if self.pos_encoder is not None:
                src_key_padding_mask_filtered = src_key_padding_mask[~empty_rois_mask]
                attended_features_filtered[~src_key_padding_mask_filtered] = self.pos_encoder(attended_features_filtered,
                                                                                              positional_input[~empty_rois_mask] if positional_input is not None else None)[~src_key_padding_mask_filtered]

            # (b, xyz, f) -> (xyz, b, f)
            attended_features_filtered = attended_features_filtered.permute(1, 0, 2)
            # (xyz, b, f) -> (b, xyz, f)
            attended_features[~empty_rois_mask] = self.transformer_encoder(attended_features_filtered,
                                                                           src_key_padding_mask=src_key_padding_mask[~empty_rois_mask]).permute(1, 0, 2).contiguous()
        else:
            if self.pos_encoder is not None:
                attended_features = self.pos_encoder(attended_features, positional_input)

            # (b, xyz, f) -> (xyz, b, f)
            attended_features = attended_features.permute(1, 0, 2)
            # (xyz, b, f) -> (b, xyz, f)
            attended_features = self.transformer_encoder(attended_features).permute(1, 0, 2).contiguous()

        return attended_features


class FrequencyPositionalEncoding3d(nn.Module):
    def __init__(self, d_model, max_spatial_shape, dropout=0.1):
        """
        Sine + Cosine positional encoding based on Attention is all you need (https://arxiv.org/abs/1706.03762) in 3D. Using the same concept as DETR,
        the sinusoidal encoding is independent across each spatial dimension.
        Args:
            d_model: Dimension of the input features. Must be divisible by 6 ((cos + sin) * 3 dimensions = 6)
            max_spatial_shape: (3,) Size of each dimension
            dropout: Dropout probability
        """
        super().__init__()

        assert len(max_spatial_shape) == 3, 'Spatial dimension must be 3'
        assert d_model % 6 == 0, f'Feature dimension {d_model} not divisible by 6'
        self.max_spatial_shape = max_spatial_shape

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([d_model] + list(max_spatial_shape))

        d_model = int(d_model / len(max_spatial_shape))

        # Equivalent to attention is all you need encoding: https://arxiv.org/abs/1706.03762
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        pos_x = torch.arange(0., max_spatial_shape[0]).unsqueeze(1)
        pos_y = torch.arange(0., max_spatial_shape[1]).unsqueeze(1)
        pos_z = torch.arange(0., max_spatial_shape[2]).unsqueeze(1)

        pe[0:d_model:2, ...] = torch.sin(pos_x * div_term).transpose(0, 1)[:, :, None, None].repeat(1, 1, max_spatial_shape[1], max_spatial_shape[2])
        pe[1:d_model:2, ...] = torch.cos(pos_x * div_term).transpose(0, 1)[:, :, None, None].repeat(1, 1, max_spatial_shape[1], max_spatial_shape[2])
        pe[d_model:2*d_model:2, ...] = torch.sin(pos_y * div_term).transpose(0, 1)[:, None, :, None].repeat(1, max_spatial_shape[0], 1, max_spatial_shape[2])
        pe[d_model+1:2*d_model:2, ...] = torch.cos(pos_y * div_term).transpose(0, 1)[:, None, :, None].repeat(1, max_spatial_shape[0], 1, max_spatial_shape[2])
        pe[2*d_model:3*d_model:2, ...] = torch.sin(pos_z * div_term).transpose(0, 1)[:, None, None, :].repeat(1, max_spatial_shape[0], max_spatial_shape[1], 1)
        pe[2*d_model+1:3*d_model:2, ...] = torch.cos(pos_z * div_term).transpose(0, 1)[:, None, None, :].repeat(1, max_spatial_shape[0], max_spatial_shape[1], 1)

        self.register_buffer('pe', pe)

    def forward(self, point_features, positional_input, grid_size=None):
        """
        Args:
            point_features: (b, xyz, f)
            positional_input: (b, xyz, 3)
        Returns:
            point_features: (b, xyz, f)
        """
        assert len(point_features.shape) == 3
        num_points = point_features.shape[1]
        num_features = point_features.shape[2]
        if grid_size == None:
            grid_size = self.max_spatial_shape
        assert num_points == grid_size.prod()

        pe =  self.pe[:, :grid_size[0], :grid_size[1], :grid_size[2]].permute(1, 2, 3, 0).contiguous().view(1, num_points, num_features)
        point_features = point_features + pe
        return self.dropout(point_features)


class FeedForwardPositionalEncoding(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_input, d_output // 2, 1),
            nn.BatchNorm1d(d_output // 2),
            nn.ReLU(d_output // 2),
            nn.Conv1d(d_output // 2, d_output, 1),
        )

    def forward(self, point_features, positional_input, grid_size=None):
        """
        Args:
            point_features: (b, xyz, f)
            local_point_locations: (b, xyz, 3)
        Returns:
            point_features: (b, xyz, f)
        """
        pos_encoding = self.ffn(positional_input.permute(0, 2, 1))
        point_features = point_features + pos_encoding.permute(0, 2, 1)
        return point_features


def get_positional_encoder(pool_cfg):
    pos_encoder = None
    attention_cfg = pool_cfg.ATTENTION
    if attention_cfg.POSITIONAL_ENCODER == 'frequency':
        pos_encoder = FrequencyPositionalEncoding3d(d_model=attention_cfg.NUM_FEATURES,
                                                    max_spatial_shape=torch.IntTensor([pool_cfg.GRID_SIZE] * 3),
                                                    dropout=attention_cfg.DROPOUT)
    elif attention_cfg.POSITIONAL_ENCODER == 'grid_points':
        pos_encoder = FeedForwardPositionalEncoding(d_input=3, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density':
        pos_encoder = FeedForwardPositionalEncoding(d_input=1, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density_grid_points':
        pos_encoder = FeedForwardPositionalEncoding(d_input=4, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density_centroid':
        pos_encoder = FeedForwardPositionalEncoding(d_input=7, d_output=attention_cfg.NUM_FEATURES)

    return pos_encoder

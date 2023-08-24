import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def load_params_from_file(model, filename, logger, to_cpu=False,
                          fix_pretrained_weights=False):

    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' %
        (filename, 'CPU' if to_cpu else 'GPU'))
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    model_state_disk = checkpoint['model_state']

    if 'version' in checkpoint:
        logger.info('==> Checkpoint trained from version: %s' %
            checkpoint['version'])

    update_model_state = {}
    for key, val in model_state_disk.items():
        if key in model.state_dict() and model.state_dict()[
            key].shape == model_state_disk[key].shape:
            update_model_state[key] = val

    state_dict = model.state_dict()
    state_dict.update(update_model_state)
    model.load_state_dict(state_dict)

    if fix_pretrained_weights:
        for name,param in model.named_parameters():
            if name in update_model_state:
                param.requires_grad=False

    for key in state_dict:
        if key not in update_model_state:
            logger.info('Not updated weight %s: %s' % (key, str(state_dict[key].shape)))

    logger.info('==> Done (loaded %d/%d)' % (len(update_model_state), len(model.state_dict())))


def load_params_with_optimizer(model, filename, to_cpu=False,
                               optimizer=None, logger=None):
    if not os.path.isfile(filename):
        raise FileNotFoundError

    logger.info('==> Loading parameters from checkpoint %s to %s' %
        (filename, 'CPU' if to_cpu else 'GPU'))
    
    loc_type = torch.device('cpu') if to_cpu else None
    checkpoint = torch.load(filename, map_location=loc_type)
    epoch = checkpoint.get('epoch', -1)
    it = checkpoint.get('it', 0.0)

    model.load_state_dict(checkpoint['model_state'])

    if optimizer is not None:
        if 'optimizer_state' in checkpoint and checkpoint['optimizer_state'] is not None:
            logger.info('==> Loading optimizer parameters from checkpoint %s to %s' %
                (filename, 'CPU' if to_cpu else 'GPU'))
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        else:
            assert filename[-4] == '.', filename
            src_file, ext = filename[:-4], filename[-3:]
            optimizer_filename = '%s_optim.%s' % (src_file, ext)
            if os.path.exists(optimizer_filename):
                optimizer_ckpt = torch.load(optimizer_filename, map_location=loc_type)
                optimizer.load_state_dict(optimizer_ckpt['optimizer_state'])

    if 'version' in checkpoint:
        logger.info('==> Checkpoint trained from version: %s' % checkpoint['version'])
    logger.info('==> Done')

    return it, epoch


def make_linear_layers(linear_cfg, input_channels, output_channels, output_use_norm=False):
    linear_layers = []
    c_in = input_channels
    for k in range(0, linear_cfg.__len__()):
        linear_layers.extend([
            nn.Linear(c_in, linear_cfg[k], bias=False),
            nn.BatchNorm1d(linear_cfg[k], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ])
        c_in = linear_cfg[k]
    if output_use_norm:
        linear_layers.append(nn.Linear(c_in, output_channels, bias=False)),
        linear_layers.append(nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01)),
        linear_layers.append(nn.ReLU()),
    else:
        linear_layers.append(nn.Linear(c_in, output_channels, bias=True))
    return nn.Sequential(*linear_layers)

def make_fc_layers(linear_cfg, input_channels, output_channels, output_use_norm=False):
    fc_layers = []
    c_in = input_channels
    for k in range(0, linear_cfg.__len__()):
        fc_layers.extend([
            nn.Conv1d(c_in, linear_cfg[k], kernel_size=1, bias=False),
            nn.BatchNorm1d(linear_cfg[k], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ])
        c_in = linear_cfg[k]
    if output_use_norm:
        fc_layers.append(nn.Conv1d(c_in, output_channels, kernel_size=1, bias=False)),
        fc_layers.append(nn.BatchNorm1d(output_channels, eps=1e-3, momentum=0.01)),
        fc_layers.append(nn.ReLU()),
    else:
        fc_layers.append(nn.Conv1d(c_in, output_channels, kernel_size=1, bias=True))
    return nn.Sequential(*fc_layers)

def make_conv_layers(conv_cfg, input_channels, output_channels, output_use_norm=False):
    conv_layers = []
    c_in = input_channels
    for k in range(0, conv_cfg.__len__()):
        conv_layers.extend([
            nn.Conv2d(c_in, conv_cfg[k], kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(conv_cfg[k], eps=1e-3, momentum=0.01),
            nn.ReLU(),
        ])
        c_in = conv_cfg[k]

    if output_use_norm:
        conv_layers.append(nn.Conv2d(c_in, output_channels, kernel_size=1, stride=1, padding=0, bias=False))
        conv_layers.append(nn.BatchNorm2d(output_channels, eps=1e-3, momentum=0.01))
        conv_layers.append(nn.ReLU())
    else:
        conv_layers.append(nn.Conv2d(c_in, output_channels, kernel_size=1, stride=1, padding=0, bias=True))
    return nn.Sequential(*conv_layers)


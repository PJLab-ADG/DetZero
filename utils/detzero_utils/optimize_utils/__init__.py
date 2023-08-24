from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper, OptimWrapper2
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY)
        
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    elif optim_cfg.OPTIMIZER == 'adamW_onecycle':
        # flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        # get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        params = list()
        add_params(params, model, optim_cfg.PARAMWISE, optim_cfg.LR, optim_cfg.WEIGHT_DECAY)
        optimizer_func = partial(optim.AdamW, betas=(0.9, 0.99))
        optimizer = OptimWrapper2.create_from_params(
            optim_cfg.LR, optimizer_func, params, wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )

    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
    def lr_lbmd(cur_epoch):
        cur_decay = 1
        for decay_step in decay_steps:
            if cur_epoch >= decay_step:
                cur_decay = cur_decay * optim_cfg.LR_DECAY
        return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    last_step = (last_epoch + 1)*total_iters_each_epoch-1
    if 'onecycle' in optim_cfg.OPTIMIZER:
        lr_scheduler = OneCycle(
            optimizer, total_steps, last_step, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_step)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler


def add_params(params, module, paramwise_cfg, base_lr, weight_decay, prefix=''):
    """Add all parameters of module to the params list.
    The parameters of the given module will be added to the list of param
    groups, with specific rules defined by paramwise_cfg.

    Args:
        params (list[dict]): A list of param groups, it will be modified
            in place.
        module (nn.Module): The module to be added.
        prefix (str): The prefix of the module
        is_dcn_module (int|float|None): If the current module is a
            submodule of DCN, `is_dcn_module` will be passed to
            control conv_offset layer's learning rate. Defaults to None.
    """
    # get param-wise options
    custom_keys = paramwise_cfg.get('custom_keys', {})
    # first sort with alphabet order and then sort with reversed len of str
    sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)    

    for name, param in module.named_parameters(recurse=False):
        param_group = {'params': [param]}
        if not param.requires_grad:
            params.append(param_group)
            continue
        # if the parameter match one of the custom keys, ignore other rules
        is_custom = False
        for key in sorted_keys:
            if key in f'{prefix}.{name}':
                is_custom = True
                lr_mult = custom_keys[key].get('lr_mult', 1.)
                param_group['lr_mult'] = lr_mult
                param_group['lr'] = base_lr * lr_mult

                if weight_decay is not None:
                    decay_mult = custom_keys[key].get('decay_mult', 1.)
                    param_group['weight_decay'] = weight_decay * decay_mult
                break
        
        if not is_custom:
            # bias_lr_mult affects all bias parameters
            # except for norm.bias dcn.conv_offset.bias
            param_group['lr'] = base_lr
            param_group['lr_mult'] = 1.
            param_group['weight_decay'] = weight_decay

        params.append(param_group)
    
    
    for child_name, child_mod in module.named_children():
            child_prefix = f'{prefix}.{child_name}' if prefix else child_name
            add_params(
                params,
                child_mod,
                paramwise_cfg,
                base_lr,
                weight_decay,
                prefix=child_prefix)

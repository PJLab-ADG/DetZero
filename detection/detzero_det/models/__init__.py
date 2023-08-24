from collections import namedtuple

import numpy as np
import torch

from .centerpoint import CenterPoint

__all__ = {
    'CenterPoint': CenterPoint
}


def build_network(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg,
        num_class=num_class,
        dataset=dataset
    )
    return model

def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if key in ['frame_id', 'metadata', 'sequence_name', 'pose', 'tta_ops',
                   'aug_matrix_inv']:
            continue
        elif isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        else:
            continue

def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func

from collections import namedtuple

import numpy as np
import torch

from .geometry_refine_model import GeometryRefineModel
from .position_refine_model import PositionRefineModel
from .confidence_refine_model import ConfidenceRefineModel

__all__ = {
    'GeometryRefineModel': GeometryRefineModel,
    'PositionRefineModel': PositionRefineModel,
    'ConfidenceRefineModel': ConfidenceRefineModel
}

def build_network(model_cfg, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, dataset=dataset
    )
    return model


def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        if key in ['frame_id', 'sequence_name', 'poses']:
            continue
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()


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

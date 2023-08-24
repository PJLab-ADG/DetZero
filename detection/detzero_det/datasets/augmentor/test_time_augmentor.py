import copy
from functools import partial

import numpy as np

from detzero_utils.common_utils import rotate_points_along_z


class TestTimeAugmentor(object):
    def __init__(self, augmentor_configs, logger=None):
        self.logger = logger

        self.tta_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.tta_queue.append(cur_augmentor)

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def flip_along_x(self, data_dict):
        data_dict_flip_x = copy.deepcopy(data_dict)
        data_dict_flip_x['points'][:, 1] = - data_dict_flip_x['points'][:, 1]
        return {"tta_flip_x": data_dict_flip_x}

    def flip_along_y(self, data_dict):
        data_dict_flip_y = copy.deepcopy(data_dict)
        data_dict_flip_y['points'][:, 0] = - data_dict_flip_y['points'][:, 0]
        return {"tta_flip_y": data_dict_flip_y}

    def flip_along_xy(self, data_dict):
        data_dict_flip_xy = copy.deepcopy(data_dict)
        data_dict_flip_xy['points'][:, 0] = - data_dict_flip_xy['points'][:, 0]
        data_dict_flip_xy['points'][:, 1] = - data_dict_flip_xy['points'][:, 1]
        return {'tta_flip_xy': data_dict_flip_xy}

    def world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_rotation, config=config)
        data_dict_list = []
        for rot_angle in config['ROT_ANGLE']:
            if rot_angle == 0.: continue
            data_dict_rot = copy.deepcopy(data_dict)
            data_dict_rot["points"] = rotate_points_along_z(data_dict_rot["points"][np.newaxis, :, :],\
                                                            np.array([rot_angle]))[0]
            key_name = 'tta_rot_%s' % str(rot_angle)
            data_dict_list.append({key_name: data_dict_rot})
        return data_dict_list

    def world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_flip, config=config)
        data_dict_list = []
        for cur_axis in config['ALONG_AXIS_LIST']:
            data_dict_flip = getattr(self, 'flip_along_%s' % cur_axis)(
                data_dict
            )
            data_dict_list.append(data_dict_flip)
        return data_dict_list

    def world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.world_scaling, config=config)
        scale_list = config['SCALE_RANGE']
        data_dict_list = []
        for scale_factor in scale_list:
            if scale_factor == 1.: continue
            data_dict_scale = copy.deepcopy(data_dict)
            data_dict_scale['points'][:, :3] *= scale_factor
            key_name = 'tta_scale_%s' % str(scale_factor)
            data_dict_list.append({key_name: data_dict_scale})
        return data_dict_list

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                ...
        Returns:
        """
        data_dict_tta = {}
        data_dict_tta['tta_original'] = data_dict
        for cur_augmentor in self.tta_queue:
            data_dict_list = cur_augmentor(data_dict=data_dict)
            for tta_dict in data_dict_list:
                data_dict_tta.update(tta_dict)

        return data_dict_tta

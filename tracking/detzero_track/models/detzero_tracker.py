from detzero_track.models import tracking_modules as tk_modules


class DetZeroTracker():
    def __init__(self, model_cfg, logger=None):
        self.model_cfg = model_cfg
        self.logger = logger
        self.module_topology = [
            'tracking', 'post_process'
        ]
        self.module_list = self.build_tracker()

    def build_tracker(self):
        model_info_dict = {
            'module_list': [],
        }
        for module_name in self.module_topology:
            module, model_info_dict = getattr(self, 'build_%s' % module_name)(
                model_info_dict=model_info_dict
            )
        return model_info_dict['module_list']

    def build_tracking(self, model_info_dict):
        if self.model_cfg.get('TRACKING', None) is None:
            return None, model_info_dict

        tracking_module = tk_modules.__all__[self.model_cfg.TRACKING.NAME](
            model_cfg=self.model_cfg.TRACKING
        )
        model_info_dict['module_list'].append(tracking_module)
        return tracking_module, model_info_dict

    def build_post_process(self, model_info_dict):
        if self.model_cfg.get('POST_PROCESS', None) is None:
            return None, model_info_dict

        post_process_module = tk_modules.__all__[self.model_cfg.POST_PROCESS.NAME](
            processor_configs=self.model_cfg.POST_PROCESS,
        )
        model_info_dict['module_list'].append(post_process_module)
        return post_process_module, model_info_dict

    def forward(self, data_dict):
        for module in self.module_list:
            data_dict = module.forward(data_dict=data_dict)
        return data_dict


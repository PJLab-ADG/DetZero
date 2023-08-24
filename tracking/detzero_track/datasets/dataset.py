import os

import torch
import numpy as np
from collections import defaultdict

from .data_processor import DataProcessor


class DatasetTemplate(torch.utils.data.Dataset):
    """
    """
    def __init__(self, dataset_cfg, det_path, split, root_path=None, logger=None):
        self.dataset_cfg = dataset_cfg
        self.det_path = det_path
        self.root_path = root_path if root_path is not None else self.dataset_cfg.DATA_PATH
        self.logger = logger
        self.split = split

        self.data_processor = DataProcessor(
            self.dataset_cfg.DATA_PROCESSOR,
            os.path.join(self.root_path, self.dataset_cfg.PROCESSED_DATA_TAG)
        )

    @property
    def assign_mode(self) -> str: 
        return False if self.split == 'test' else True


if __name__ == "__main__":
    pass

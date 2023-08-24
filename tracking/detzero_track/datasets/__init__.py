import torch
from torch.utils.data import DataLoader

from .waymo_dataset import WaymoTrackDataset

__all__ = {
    "WaymoTrackDataset": WaymoTrackDataset
}


def build_dataloader(dataset_cfg, log_time, data_path, batch_size, 
                     workers, split, logger, root_path=None):
    dataset = __all__[dataset_cfg.DATASET](
        dataset_cfg=dataset_cfg,
        data_path=data_path,
        split=split,
        root_path=root_path,
        log_time=log_time,
        logger=logger
    )
    # init as spawn mode
    torch.multiprocessing.set_start_method('spawn')

    dataloader = DataLoader(
        dataset, batch_size=batch_size, pin_memory=True, num_workers=workers,
        shuffle=False, collate_fn=dataset.collate_batch, drop_last=False, timeout=0
    )

    return dataset, dataloader

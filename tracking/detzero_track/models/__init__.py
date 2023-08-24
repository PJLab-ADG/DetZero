import pickle

from tqdm import tqdm
from functools import partial

from detzero_utils.common_utils import get_log_info, multi_processing

from .detzero_tracker import DetZeroTracker

__all__ = {
    'DetZeroTracker': DetZeroTracker
}


def build_model(model_cfg, logger=None):
    model=__all__[model_cfg.NAME](
        model_cfg=model_cfg,
        logger=logger,
    )
    return model

def run_model(model, dataloader, dataset, workers, cfgs=None, logger=None):
    if logger is not None:
        logger.info(get_log_info('Running Tracking Module!'))

    if dataset.assign_mode:
        from detzero_track.models.tracking_modules import target_assign
        assigner = partial(
            target_assign.assign_track_target, iou_thresholds=cfgs.REFINING.IOU_THRESHOLDS
        )
        assign_data = dict()

    track_data = dict()
    drop_data = dict()
    total_iter = len(dataloader)
    pbar = tqdm(dataloader, total=total_iter, ascii=True, ncols=140)
    for idx, (seq_names, data_dicts) in enumerate(pbar):
        model_outputs = multi_processing(
            function=model.forward, 
            data_dict=data_dicts['detection'], 
            workers=workers
        )
        track_data.update(dict(zip(seq_names, model_outputs)))

        if dataset.assign_mode:
            input_data = list(zip(data_dicts['detection'], model_outputs, data_dicts['gt']))
            refine_outputs = multi_processing(assigner, input_data, workers)
            assign_data.update(dict(zip(seq_names, refine_outputs)))
        drop_data.update(dict(zip(seq_names, data_dicts['det_drop'])))

    track_path = dataset.get_track_path()
    with open(track_path, 'wb') as f:
        if dataset.assign_mode:
            pickle.dump(assign_data, f)
        else:
            pickle.dump(track_data, f)

    drop_path = dataset.get_drop_path()
    with open(drop_path, 'wb') as f:
        pickle.dump(drop_data, f)

    if logger is not None:
        logger.info(get_log_info('Tracking module running finished!'))

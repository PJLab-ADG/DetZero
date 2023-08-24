import os 
import pickle
import time

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter

from detzero_utils import common_utils

from detzero_refine.models import load_data_to_gpu


def statistics_info(cfg, ret_dict, metric, disp_dict):
    if cfg.MODEL.POST_PROCESSING.get('GENERATE_RECALL', True):
        for cur_th in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
            metric['Num@box input %s' % str(cur_th)] += \
                ret_dict['Box level input %s' % str(cur_th)]
            metric['Num@box output %s' % str(cur_th)] += \
                ret_dict['Box level output %s' % str(cur_th)]

            metric['Num@box input (static) %s' % str(cur_th)] += \
                ret_dict['Box level input (static) %s' % str(cur_th)]
            metric['Num@box output (static) %s' % str(cur_th)] += \
                ret_dict['Box level output (static) %s' % str(cur_th)]
            metric['Num@box input (dynamic) %s' % str(cur_th)] += \
                ret_dict['Box level input (dynamic) %s' % str(cur_th)]
            metric['Num@box output (dynamic) %s' % str(cur_th)] += \
                ret_dict['Box level output (dynamic) %s' % str(cur_th)]
            
            metric['Num@track input %s' % str(cur_th)] += \
                ret_dict['Track level input %s' % str(cur_th)]
            metric['Num@track output %s' % str(cur_th)] += \
                ret_dict['Track level output %s' % str(cur_th)]

            metric['Num@track input (static) %s' % str(cur_th)] += \
                ret_dict['Track level input (static) %s' % str(cur_th)]
            metric['Num@track output (static) %s' % str(cur_th)] += \
                ret_dict['Track level output (static) %s' % str(cur_th)]
            metric['Num@track input (dynamic) %s' % str(cur_th)] += \
                ret_dict['Track level input (dynamic) %s' % str(cur_th)]
            metric['Num@track output (dynamic) %s' % str(cur_th)] += \
                ret_dict['Track level output (dynamic) %s' % str(cur_th)]

        metric['Box num'] += ret_dict.get('Box num', 0)
        metric['Track num'] += ret_dict.get('Track num', 0)
        metric['static_num'] += ret_dict.get('static', 0)
        metric['dynamic_num'] += ret_dict.get('dynamic', 0)
        metric['matched_up'] += ret_dict.get('matched_up', 0)
        metric['matched_down'] += ret_dict.get('matched_down', 0)
        metric['unmatched_down'] += ret_dict.get('unmatched_down', 0)
        metric['unmatched_up'] += ret_dict.get('unmatched_up', 0)

        metric['gt_score'].extend(ret_dict.get('gt_score', []))
        metric['pred_score'].extend(ret_dict.get('pred_score', []))
        metric['ori_score'].extend(ret_dict.get('ori_score', []))

        min_th = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
        disp_dict['Recall@box_%s' % str(min_th)] = '(%d, %d) / %d' % (\
            metric['Num@box input %s' % str(min_th)],
            metric['Num@box output %s' % str(min_th)],
            metric['Box num']
        )
        disp_dict['Recall@track_%s' % str(min_th)] = '(%d, %d) / %d' % (\
            metric['Num@track input %s' % str(min_th)],
            metric['Num@track output %s' % str(min_th)],
            metric['Track num']
        )

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    if result_dir is not None:
        result_dir.mkdir(parents=True, exist_ok=True)

    print(result_dir, save_to_file)

    accumulated_iter = 0

    metric = {
        'Box num': 0,
        'Track num': 0,
        'static_num': 0,
        'dynamic_num': 0,
        'matched_up': 0,
        'matched_down': 0,
        'unmatched_down': 0,
        'unmatched_up': 0,
        'gt_score': [],
        'pred_score': [],
        'ori_score': [],
        'acc@0.75': 0,
    }
    for cur_th in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric.update({
            'Num@box input %s' % str(cur_th): 0,
            'Num@box output %s' % str(cur_th): 0,
            'Num@box input (static) %s' % str(cur_th): 0,
            'Num@box output (static) %s' % str(cur_th): 0,
            'Num@box input (dynamic) %s' % str(cur_th): 0,
            'Num@box output (dynamic) %s' % str(cur_th): 0,
            'Num@track input %s' % str(cur_th): 0,
            'Num@track output %s' % str(cur_th): 0,
            'Num@track input (static) %s' % str(cur_th): 0,
            'Num@track output (static) %s' % str(cur_th): 0,
            'Num@track input (dynamic) %s' % str(cur_th): 0,
            'Num@track output (dynamic) %s' % str(cur_th): 0
        })

    dataset = dataloader.dataset
    det_annos = []

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            broadcast_buffers=False
        )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    save_dict = {}
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict, tb_dict_val = model(batch_dict)
        disp_dict = {}
        statistics_info(cfg, ret_dict, metric, disp_dict)
        if save_to_file:
            annos = dataset.generate_prediction_dicts(
                batch_dict, pred_dicts, save_dict
            )
            det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    logger.info('Num of Object boxes: %s' % metric['Box num'])
    logger.info('Num of Object tracks: %s' % metric['Track num'])

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')
        save_dict = common_utils.merge_results_dist([save_dict], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if save_to_file:
        if dist_test:
            for key in save_dict[0].keys():
                for k in range(1, world_size):
                    save_dict[0][key].update(save_dict[k][key])
            save_dict = save_dict[0]

        # specify the file name format
        if cfg.MODEL.NAME == 'GeometryRefineModel':
            save_name = 'geometry'
        elif cfg.MODEL.NAME == 'PositionRefineModel':
            save_name = 'position'
        elif cfg.MODEL.NAME == 'ConfidenceRefineModel':
            save_name = 'confidence'

        if len(cfg.CLASS_NAMES) == 1:
            save_name = cfg.CLASS_NAMES[0] + '_%s' % save_name
        elif len(cfg.CLASS_NAMES) == 3:
            save_name = 'Cyclist_%s' % save_name

        save_path = os.path.join(result_dir, '%s_%s.pkl'
            % (save_name, str(result_dir).split('/')[-1]))

        with open(save_path, 'wb') as f:
            pickle.dump(save_dict, f)
        logger.info('Result is save to %s' % save_path)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_box_num = metric['Box num']
    gt_tk_num = metric['Track num']
    for cur_th in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        rec_box_in = metric['Num@box input %s' % str(cur_th)] / max(gt_box_num, 1)
        rec_box_out = metric['Num@box output %s' % str(cur_th)] / max(gt_box_num, 1)
        rec_tk_in = metric['Num@track input %s' % str(cur_th)] / max(gt_tk_num, 1)
        rec_tk_out = metric['Num@track output %s' % str(cur_th)] / max(gt_tk_num, 1)

        logger.info('Num@box input %s: %f' % (cur_th, metric['Num@box input %s' % cur_th]))
        logger.info('Num@box output %s: %f' % (cur_th, metric['Num@box output %s' % cur_th]))
        logger.info('Recall@box input %s: %f' % (cur_th, rec_box_in))
        logger.info('Recall@box output %s: %f' % (cur_th, rec_box_out))
        logger.info('====================================================')
        logger.info('Num@box input (static) %s: %f' % (cur_th, metric['Num@box input (static) %s' % cur_th]))
        logger.info('Num@box output (static) %s: %f' % (cur_th, metric['Num@box output (static) %s' % cur_th]))
        logger.info('Recall@box input (static) %s: %f' % (cur_th, metric['Num@box input (static) %s' % cur_th] / max(metric['static_num'], 1)))
        logger.info('Recall@box output (static) %s: %f' % (cur_th, metric['Num@box output (static) %s' % cur_th] / max(metric['static_num'], 1)))
        logger.info('====================================================')
        logger.info('Num@box input (dynamic) %s: %f' % (cur_th, metric['Num@box input (dynamic) %s' % cur_th]))
        logger.info('Num@box output (dynamic) %s: %f' % (cur_th, metric['Num@box output (dynamic) %s' % cur_th]))
        logger.info('Recall@box input (dynamic) %s: %f' % (cur_th, metric['Num@box input (dynamic) %s' % cur_th] / max(metric['dynamic_num'], 1)))
        logger.info('Recall@box output (dynamic) %s: %f' % (cur_th, metric['Num@box output (dynamic) %s' % cur_th] / max(metric['dynamic_num'], 1)))
        logger.info('====================================================')
        logger.info('Num@track input %s: %f' % (cur_th, metric['Num@track input %s' % cur_th]))
        logger.info('Num@track output %s: %f' % (cur_th, metric['Num@track output %s' % cur_th]))
        logger.info('Recall@track input %s: %f' % (cur_th, rec_box_in))
        logger.info('Recall@track output %s: %f' % (cur_th, rec_box_out))
        logger.info('====================================================')
        logger.info('Num@track input (static) %s: %f' % (cur_th, metric['Num@track input (static) %s' % cur_th]))
        logger.info('Num@track output (static) %s: %f' % (cur_th, metric['Num@track output (static) %s' % cur_th]))
        logger.info('Num@track input (dynamic) %s: %f' % (cur_th, metric['Num@track input (dynamic) %s' % cur_th]))
        logger.info('Num@track output (dynamic) %s: %f' % (cur_th, metric['Num@track input (dynamic) %s' % cur_th]))
        logger.info('====================================================')
        logger.info('Box num (static): {0}, (dynamic): {1}'.format(metric['static_num'], metric['dynamic_num']))
        logger.info('Matched gt-box num: {0}, matched gt-track num: {1}'.format(gt_box_num, gt_tk_num))
        logger.info('Matched increase: {0}'.format(metric['matched_up']))
        logger.info('Matched decrease: {0}'.format(metric['matched_down']))
        logger.info('Unmatched increase: {0}'.format(metric['unmatched_up']))
        logger.info('Unmatched decrease: {0}'.format(metric['unmatched_down']))
        logger.info('====================================================')

        ret_dict['Recall@box/input_%s' % str(cur_th)] = rec_box_in
        ret_dict['Recall@box/output_%s' % str(cur_th)] = rec_box_out
        ret_dict['Recall@track/input_%s' % str(cur_th)] = rec_tk_in
        ret_dict['Recall@track/output_%s' % str(cur_th)] = rec_tk_out

    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass

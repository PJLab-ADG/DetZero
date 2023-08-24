import copy
import argparse
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from tabulate import tabulate

from detzero_utils import common_utils
from detzero_det.datasets.waymo.waymo_eval_detection import WaymoDetectionMetricsEstimator

from waymo_eval_tracking import WaymoTrackingMetricsEstimator

HUMAN_STUDY_LIST = [
    '17703234244970638241_220_000_240_000',
    '15611747084548773814_3740_000_3760_000',
    '11660186733224028707_420_000_440_000',
    '1024360143612057520_3580_000_3600_000',
    '6491418762940479413_6520_000_6540_000',
]


def parse_config():
    parser = argparse.ArgumentParser(description='Offline evaluation tool of DetZero.')
    parser.add_argument('--det_result_path', type=str, default='', help='path to the prediction result file')
    parser.add_argument('--gt_info_path', type=str, default='../data/waymo/waymo_infos_val.pkl',
                        help='path to ground-truth info pkl file')
    parser.add_argument('--evaluate_metrics', nargs='+', default=['object'],
                        help='metrics that used in evaluation. support multiple (object, range)')
    parser.add_argument('--iou_type', type=str, default='3d', help='support 3d or bev for matching boxes')
    parser.add_argument('--class_name', nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'],
                        help='The class names of the detection, default=["Vehicle", "Pedestrian", "Cyclist"]')
    parser.add_argument('--info_with_fakelidar', action='store_true', default=False)
    parser.add_argument('--distance_thresh', type=int, default=1000)
    parser.add_argument('--tracking', action="store_true", default=False, help="evaluate tracking performance")
    parser.add_argument('--human_study', action='store_true', default=False)

    args = parser.parse_args()
    log_file = Path(args.det_result_path).resolve().parent / 'mAP.txt'
    logger = common_utils.create_logger(log_file)

    return args, logger


def main():
    args, logger = parse_config()
    if any([item not in ['object', 'range'] for item in args.evaluate_metrics]):
        raise ValueError('evaluate_metrics error')

    logger.info(args)

    logger.info('Loading ground-truth infos: ' + args.gt_info_path)
    with open(args.gt_info_path, 'rb') as f:
        gt_infos = pickle.load(f)
    gt_infos_table = defaultdict(dict)
    missed_info_table = defaultdict(dict)

    # prepare gt items
    for item in gt_infos:
        if args.human_study:
            if item['sequence_name'] not in HUMAN_STUDY_LIST: continue
        
        gt_infos_table[item['sequence_name']][item['sample_idx']] = item
        missed_info_table[item['sequence_name']][item['sample_idx']] = item

    det_result = pickle.load(open(args.det_result_path, 'rb'))
    logger.info("Prediction Set Length: {}, Ground Truth Set Length: {}.".format(
        len(det_result), len(gt_infos)))

    logger.info('Generating evaluation data pair (det, gt)')
    eval_gt_annos = []
    eval_det_annos = []

    # prepare predicted items
    for item in det_result:
        if args.human_study:
            if item['sequence_name'] not in HUMAN_STUDY_LIST: continue

        eval_det_annos.append(copy.deepcopy(item))
        if not args.tracking:
            eval_gt_annos.append(copy.deepcopy(gt_infos_table[item['sequence_name']][item['frame_id']]['annos']))
        else:
            eval_gt_annos.append(copy.deepcopy(gt_infos_table[item['sequence_name']][item['frame_id']]))
        del missed_info_table[item['sequence_name']][item['frame_id']]

    # process the frames that don't contain predicted results
    if not len(det_result) == len(gt_infos):
        logger.info("All frames must be detected, empty list will be appended.")

        for k in missed_info_table.keys():
            for f in missed_info_table[k].keys():
                logger.info("sequence:{}, frame:{} is missed".format(k, f))
                empty = {
                    'sequence_name': k,
                    'frame_id': f,
                    'boxes_lidar': np.array([]).reshape(0, 7),
                    'score': np.array([]),
                    'name': np.array([])
                }
                eval_gt_annos.append(missed_info_table[k][f]['annos'])
                eval_det_annos.append(empty)

        logger.info("After modification | Prediction Set Length: {}, Ground Truth Set Length: {}."
            .format(len(eval_gt_annos), len(eval_det_annos)))

    logger.info('\tEvaluation data pair: ' + str(len(eval_gt_annos)))

    logger.info('Start Evaluation')
    if not args.tracking:
        if args.iou_type == '3d' or args.iou_type == 'bev':
            args.evaluate_metrics.append(args.iou_type)
            evaluator = WaymoDetectionMetricsEstimator()
        else:
            raise KeyError
    else:
        evaluator = WaymoTrackingMetricsEstimator()

    ap_dict = evaluator.waymo_evaluation(
        eval_det_annos,
        eval_gt_annos,
        config_type=args.evaluate_metrics,
        class_name=args.class_name,
        distance_thresh=args.distance_thresh,
        fake_gt_infos=args.info_with_fakelidar
    )

    logger.info('================= Evaluation Result =================')
    
    logger.info('')
    logger.info(args.det_result_path)

    ap_result_str = '\n'
    new_ap_dict = dict()
    for key in ap_dict:
        new_ap_dict[key] = ap_dict[key][0]
        ap_result_str += '%s: %.4f \n' % (key, new_ap_dict[key])
    logger.info(ap_result_str)

    table_header = ['Category', 'L1/AP', 'L1/APH', 'L2/AP', 'L2/APH']
    table_data = []

    if 'object' in args.evaluate_metrics:
        if not args.tracking:
            table_data.extend([
                ('Vehicle', ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/APH'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/APH'][0]),
                ('Pedestrain', ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/AP'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/APH'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/AP'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/APH'][0]),
                ('Cyclist', ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/AP'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/APH'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/AP'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/APH'][0]),
                ('Sign', ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/AP'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/APH'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/AP'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/APH'][0])
            ])
        else:
            table_header = ['Category', 'MOTA', 'MOTP', 'MISS', 'MISMATCH', 'FP']
            table_data.extend([
                ('Vehicle', ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/MOTA'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/MOTP'][0], 
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/MISS'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/MISMATCH'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/FP'][0]),
                ('Vehicle', ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/MOTA'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/MOTP'][0], 
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/MISS'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/MISMATCH'][0],
                            ap_dict['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/FP'][0]),
                ('Pedestrain', ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/MOTA'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/MOTP'][0], 
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/MISS'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/MISMATCH'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_1/FP'][0]),
                ('Pedestrain', ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/MOTA'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/MOTP'][0], 
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/MISS'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/MISMATCH'][0],
                               ap_dict['OBJECT_TYPE_TYPE_PEDESTRIAN_LEVEL_2/FP'][0]),
                ('Cyclist', ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/MOTA'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/MOTP'][0], 
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/MISS'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/MISMATCH'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_1/FP'][0]),
                ('Cyclist', ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/MOTA'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/MOTP'][0], 
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/MISS'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/MISMATCH'][0],
                            ap_dict['OBJECT_TYPE_TYPE_CYCLIST_LEVEL_2/FP'][0]),
                ('Sign', ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/MOTA'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/MOTP'][0], 
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/MISS'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/MISMATCH'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_1/FP'][0]),
                ('Sign', ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/MOTA'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/MOTP'][0], 
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/MISS'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/MISMATCH'][0],
                         ap_dict['OBJECT_TYPE_TYPE_SIGN_LEVEL_2/FP'][0]),
            ])
    
    if 'range' in args.evaluate_metrics:
        table_data.extend([
            ('Vehicle_[0, 30)', ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/AP'][0],
                                ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_1/APH'][0],
                                ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/AP'][0],
                                ap_dict['RANGE_TYPE_VEHICLE_[0, 30)_LEVEL_2/APH'][0]),
            ('Vehicle_[30, 50)', ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/AP'][0],
                                 ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_1/APH'][0],
                                 ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/AP'][0],
                                 ap_dict['RANGE_TYPE_VEHICLE_[30, 50)_LEVEL_2/APH'][0]),
            ('Vehicle_[50, +inf)', ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/AP'][0],
                                   ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_1/APH'][0],
                                   ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/AP'][0],
                                   ap_dict['RANGE_TYPE_VEHICLE_[50, +inf)_LEVEL_2/APH'][0]),
            ('Pedestrian_[0, 30)', ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/AP'][0],
                                   ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_1/APH'][0],
                                   ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/AP'][0],
                                   ap_dict['RANGE_TYPE_PEDESTRIAN_[0, 30)_LEVEL_2/APH'][0]),
            ('Pedestrian_[30, 50)', ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/AP'][0],
                                    ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_1/APH'][0],
                                    ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/AP'][0],
                                    ap_dict['RANGE_TYPE_PEDESTRIAN_[30, 50)_LEVEL_2/APH'][0]),
            ('Pedestrian_[50, +inf)', ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/AP'][0],
                                      ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_1/APH'][0],
                                      ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/AP'][0],
                                      ap_dict['RANGE_TYPE_PEDESTRIAN_[50, +inf)_LEVEL_2/APH'][0]),
            ('Cyclist_[0, 30)', ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/AP'][0],
                                ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_1/APH'][0],
                                ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/AP'][0],
                                ap_dict['RANGE_TYPE_CYCLIST_[0, 30)_LEVEL_2/APH'][0]),
            ('Cyclist_[30, 50)', ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/AP'][0],
                                 ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_1/APH'][0],
                                 ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/AP'][0],
                                 ap_dict['RANGE_TYPE_CYCLIST_[30, 50)_LEVEL_2/APH'][0]),
            ('Cyclist_[50, +inf)', ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/AP'][0],
                                   ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_1/APH'][0],
                                   ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/AP'][0],
                                   ap_dict['RANGE_TYPE_CYCLIST_[50, +inf)_LEVEL_2/APH'][0]),
            ('Sign_[0, 30)', ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_1/AP'][0],
                             ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_1/APH'][0],
                             ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_2/AP'][0],
                             ap_dict['RANGE_TYPE_SIGN_[0, 30)_LEVEL_2/APH'][0]),
            ('Sign_[30, 50)', ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_1/AP'][0],
                              ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_1/APH'][0],
                              ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_2/AP'][0],
                              ap_dict['RANGE_TYPE_SIGN_[30, 50)_LEVEL_2/APH'][0]),
            ('Sign_[50, +inf)', ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_1/AP'][0],
                                ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_1/APH'][0],
                                ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2/AP'][0],
                                ap_dict['RANGE_TYPE_SIGN_[50, +inf)_LEVEL_2/APH'][0]),
        ])

    _ = [logger.info(line) for line in tabulate(table_data, headers=table_header, tablefmt='grid').splitlines()]

    logger.info('The whole evaulation process is done.')


if __name__ == '__main__':
    main()

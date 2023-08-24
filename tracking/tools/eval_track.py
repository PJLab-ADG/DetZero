import time
import argparse
from pathlib import Path

from detzero_utils.common_utils import create_logger, get_log_info

from detzero_track.utils.track_recall import TrackRecall


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--data_path', type=str, help='load the path of eval data')
    parser.add_argument('--class_names', type=str, nargs='+', default=['Vehicle', 'Pedestrian', 'Cyclist'], help='Class to be evaluated')
    parser.add_argument('--difficulty', type=str, nargs='+', default=['l1', 'l2'],help='Difficulty for gt objects')
    parser.add_argument('--iou_threshold', type=float, nargs='+', default=[0.7, 0.5, 0.5],help='IoU threshold for calculate TP')
    parser.add_argument('--method', type=str, default='3d', help='matching method')
    parser.add_argument('--split', type=str, default='val', help='the split')
    parser.add_argument('--workers', type=int, default=1, help='num of cpu for running evaluation module')    
    args = parser.parse_args()

    root_path = Path(__file__).resolve().parent.parent.resolve()
    return args, root_path

def main():
    args, root_path = parse_config()

    logger = create_logger()
    logger.info(get_log_info('DetZero Tracking Evaluation for Track-level Recall'))

    for key, val in vars(args).items():
        logger.info(f'{key}: {val}')

    evaluation = TrackRecall(
        root_path=root_path,
        data_path=args.data_path,
        split=args.split,
        workers=args.workers,
        class_names=args.class_names,
        difficultys=args.difficulty,
        iou_threshold=args.iou_threshold,
        method=args.method,
        logger=logger
    )
    evaluation.get_tracklet_recall()

if __name__ == '__main__':
    main()

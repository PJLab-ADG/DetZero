import os
import pickle
from pathlib import Path
import multiprocessing
import argparse
import yaml
from easydict import EasyDict

from tqdm import tqdm
import concurrent.futures as futures
from functools import partial
import numpy as np
import torch

from detzero_utils import common_utils, box_utils
from detzero_utils.ops.roiaware_pool3d import roiaware_pool3d_utils

from detzero_det.datasets.waymo.waymo_dataset import WaymoDetectionDataset
from detzero_det.datasets.waymo import waymo_utils


def get_infos_worker(save_path, sample_sequence_file_list,
                     num_workers=1, has_label=True):
    """
    Generate points npy files and infomation files from raw tfrecord data
    """
    print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
          % (1, len(sample_sequence_file_list)))
    process_single_sequence = partial(
        waymo_utils.process_single_sequence_and_save,
        save_path=save_path,
        has_label=has_label,
    )

    with futures.ThreadPoolExecutor(num_workers) as executor:
        sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
                                   total=len(sample_sequence_file_list)))
    
    all_sequences_infos = [item for infos in sequence_infos for item in infos]
    return all_sequences_infos

def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='val', processed_data_tag='waymo_processed_data',
                       workers=1):
    dataset = WaymoDetectionDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=str(data_path),
        training=False,
        logger=common_utils.create_logger()
    )
    sweeps = dataset.sweep_count
    
    train_split, val_split, test_split = 'train', 'val', 'test'
    train_filename = os.path.join(save_path, ('waymo_infos_%s.pkl' % train_split))
    val_filename = os.path.join(save_path, ('waymo_infos_%s.pkl' % val_split))
    test_filename = os.path.join(save_path, ('waymo_infos_%s.pkl' % test_split))
   
    print('============ Start to generate data infos ============')

    dataset.set_split(train_split)
    raw_data_path = os.path.join(data_path, raw_data_tag)
    
    # check whether the file exists
    sample_sequence_file_list = [
        dataset.check_sequence_name_with_all_version(os.path.join(raw_data_path, sequence_file))
        for sequence_file in dataset.sample_sequence_list
    ]
    
    waymo_infos_train = get_infos_worker(
        save_path=os.path.join(save_path, processed_data_tag),
        sample_sequence_file_list=sample_sequence_file_list,
        num_workers=workers,
        has_label=True
    )

    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    # pre-process valdation part
    dataset.set_split(val_split)
    raw_data_path = os.path.join(data_path, raw_data_tag)

    sample_sequence_file_list = [
        dataset.check_sequence_name_with_all_version(os.path.join(raw_data_path, sequence_file))
        for sequence_file in dataset.sample_sequence_list
    ]

    waymo_infos_val = get_infos_worker(
        save_path=os.path.join(save_path, processed_data_tag),
        sample_sequence_file_list=sample_sequence_file_list,
        num_workers=workers,
        has_label=True
    )

    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    dataset.set_split(test_split)
    raw_data_path = os.path.join(data_path, raw_data_tag)
    
    # check whether the file exists
    sample_sequence_file_list = [
        dataset.check_sequence_name_with_all_version(os.path.join(raw_data_path, sequence_file))
        for sequence_file in dataset.sample_sequence_list
    ]
    
    waymo_infos_test = get_infos_worker(
        save_path=os.path.join(save_path, processed_data_tag),
        sample_sequence_file_list=sample_sequence_file_list,
        num_workers=workers,
        has_label=True
    )

    with open(test_filename, 'wb') as f:
        pickle.dump(waymo_infos_test, f)
    print('----------------Waymo info test file is saved to %s----------------' % test_filename)

    print('============ Generate data infos finished ============')


def create_groundtruth_database(dataset, info_path, save_path, split='train', used_classes=None,
                                sampled_interval=1, sweep_count=[0,0], processed_data_tag=None):
    database_save_path = os.path.join(save_path, ('gt_database_%s_sampled_%d_sweep_%d'
                                % (split, sampled_interval, sweep_count[1]-sweep_count[0]+1)))
    db_info_save_path = os.path.join(save_path, ('waymo_dbinfos_%s_sampled_%d_sweep_%d.pkl'
                        % (split, sampled_interval, sweep_count[1]-sweep_count[0]+1)))

    if not os.path.exists(database_save_path):
        os.makedirs(database_save_path)

    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    all_db_infos = {}
    for k in range(0, len(infos), sampled_interval):
        print('gt_database sample: %d/%d' % (k + 1, len(infos)))
        info = infos[k]

        sequence_name = info['sequence_name']
        sample_idx = info['sample_idx']
        target_idx_list = dataset.get_sweep_idxs(info, sweep_count, k)
        target_infos, points = dataset.get_infos_and_points(target_idx_list)
        points = dataset.merge_sweeps(info, target_infos, points)

        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        gt_boxes = annos['gt_boxes_lidar']

        if k % 4 != 0 and len(names) > 0:
            mask = (names == 'Vehicle')
            names = names[~mask]
            difficulty = difficulty[~mask]
            gt_boxes = gt_boxes[~mask]

        if k % 2 != 0 and len(names) > 0:
            mask = (names == 'Pedestrian')
            names = names[~mask]
            difficulty = difficulty[~mask]
            gt_boxes = gt_boxes[~mask]

        box_idxs = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
        ).long().squeeze(dim=0).cpu().numpy()

        # MODIFICATION: Crop the points with the boxes seuqnce for mutiple sweeps
        num_obj = gt_boxes.shape[0]
        for i in range(num_obj):
            filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
            filepath = os.path.join(database_save_path, filename)

            if os.path.exists(filepath):
                print('Skip files since it has been processed before: %s' % filepath)
                continue

            # get points belonging to the object
            gt_points = points[box_idxs == i]
            gt_points[:, :3] -= gt_boxes[i, :3]

            if (used_classes is None) or names[i] in used_classes:
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)
                db_path = str(Path(filepath).relative_to(dataset.root_path))  # gt_database/xxxxx.bin

                db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                           'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                           'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
    
    for k, v in all_db_infos.items():
        print('Database %s: %d' % (k, len(v)))

    with open(db_info_save_path, 'wb') as f:
        pickle.dump(all_db_infos, f)

def create_waymo_database(dataset_cfg, class_names, data_path, save_path, sampled_interval):
    dataset = WaymoDetectionDataset(
        dataset_cfg=dataset_cfg,
        class_names=class_names,
        root_path=str(data_path),
        training=False,
        logger=common_utils.create_logger()
    )
    train_split, val_split = 'train', 'val'
    sweeps = dataset_cfg.SWEEP_COUNT

    train_filename = os.path.join(save_path, ('waymo_infos_%s.pkl' % train_split))

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    create_groundtruth_database(
        dataset=dataset,
        info_path=train_filename,
        save_path=save_path,
        split='train',
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'],
        sampled_interval=sampled_interval,
        sweep_count=dataset_cfg.SWEEP_COUNT
    )
    print('---------------Data preparation Done---------------')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    args = parser.parse_args()
   
    if args.func == 'create_waymo_infos':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file), Loader=yaml.FullLoader))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../../').resolve()
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=os.path.join(str(ROOT_DIR), 'data', 'waymo'),
            save_path=os.path.join(str(ROOT_DIR), 'data', 'waymo'),
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG,
            workers=multiprocessing.cpu_count()
        )
    
    if args.func == 'create_waymo_database':
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file), Loader=yaml.FullLoader))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../../').resolve()
        sampled_intervals = dataset_cfg.SAMPLED_INTERVAL
        create_waymo_database(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=os.path.join(str(ROOT_DIR), 'data', 'waymo'),
            save_path=os.path.join(str(ROOT_DIR), 'data', 'waymo'),
            sampled_interval=sampled_intervals['train'],
        )
    
import os
import pickle
import multiprocessing
from pathlib import Path

import numpy as np
import tensorflow as tf

from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

from detzero_utils import common_utils, box_utils

try:
    tf.enable_eager_execution()
except:
    pass

WAYMO_CLASSES = ['unknown', 'Vehicle', 'Pedestrian', 'Sign', 'Cyclist']


def generate_labels(frame, filter_empty_obj=True):
    obj_name, difficulty, dimensions, locations, heading_angles, velocities = [], [], [], [], [], []
    tracking_difficulty, speeds, accelerations, obj_ids = [], [], [], []
    num_points_in_gt = []
    laser_labels = frame.laser_labels
    for i in range(len(laser_labels)):
        box = laser_labels[i].box
        class_ind = laser_labels[i].type
        loc = [box.center_x, box.center_y, box.center_z]
        metadata = laser_labels[i].metadata
        velocity = [metadata.speed_x, metadata.speed_y]
        heading_angles.append(box.heading)
        obj_name.append(WAYMO_CLASSES[class_ind])
        difficulty.append(laser_labels[i].detection_difficulty_level)
        tracking_difficulty.append(laser_labels[i].tracking_difficulty_level)
        dimensions.append([box.length, box.width, box.height])  # lwh in unified coordinate
        locations.append(loc)
        velocities.append(velocity)
        obj_ids.append(laser_labels[i].id)
        num_points_in_gt.append(laser_labels[i].num_lidar_points_in_box)

    annotations = {}
    annotations['name'] = np.array(obj_name)
    annotations['difficulty'] = np.array(difficulty)
    annotations['dimensions'] = np.array(dimensions)
    annotations['location'] = np.array(locations)
    annotations['heading_angles'] = np.array(heading_angles)
    annotations['velocity'] = np.array(velocities)

    annotations['obj_ids'] = np.array(obj_ids)
    annotations['tracking_difficulty'] = np.array(tracking_difficulty)
    annotations['num_points_in_gt'] = np.array(num_points_in_gt)

    annotations = common_utils.drop_info_with_name(annotations, name='unknown')
    if annotations['name'].__len__() > 0:
        gt_boxes_lidar = np.concatenate([
            annotations['location'], annotations['dimensions'], annotations['heading_angles'][..., np.newaxis], annotations['velocity']],
            axis=1
        )
        if filter_empty_obj:
            mask_not_zero = (annotations['num_points_in_gt'] > 0).reshape(-1)
            annotations['name'] = annotations['name'][mask_not_zero]
            annotations['difficulty'] = annotations['difficulty'][mask_not_zero]
            annotations['dimensions'] = annotations['dimensions'][mask_not_zero, :]
            annotations['location'] = annotations['location'][mask_not_zero, :]
            annotations['heading_angles'] = annotations['heading_angles'][mask_not_zero]
            annotations['obj_ids'] = annotations['obj_ids'][mask_not_zero]
            annotations['tracking_difficulty'] = annotations['tracking_difficulty'][mask_not_zero]
            annotations['num_points_in_gt'] = annotations['num_points_in_gt'][mask_not_zero]
            gt_boxes_lidar = gt_boxes_lidar[mask_not_zero, :]
    else:
        gt_boxes_lidar = np.zeros((0, 9))
    annotations['gt_boxes_lidar'] = gt_boxes_lidar
    return annotations


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)
    
    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []

        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        
            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation


def process_single_sequence_and_save(sequence_file: str, save_path: Path, has_label:bool=True,
                                     pool:multiprocessing.Pool=None):
    """ Process single sequence and save the result: infos -> .pkl, pts -> .npy (for each frame)

    Attributes:
        sequence_file: The path of tfrecord that to be processed
        save_path: The path to save the preprocessing result
        has_label: True if tfrecord include grount-truth labels (trainval set), False otherwise (test set or new sequence)
        pool: The multiprocessing pool. leave it blank if you dont want to use multiprocessing.

    Returns:
        There is not returns when we only want to save the preprocessed result to .pkl and .npy files.
    """
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
    sequence_file_tfrecord = sequence_file[:-9] + '_with_camera_labels.tfrecord'
    
    if not os.path.exists(sequence_file):
        print('TFRecord not exists: %s' % sequence_file)
        return []

    cur_save_dir = os.path.join(save_path, sequence_name)
    if not os.path.exists(cur_save_dir):
        os.makedirs(cur_save_dir)
    pkl_file = os.path.join(cur_save_dir, ('%s.pkl' % sequence_name))

    if pkl_file.exists():
        sequence_infos = pickle.load(open(pkl_file, 'rb'))
        print('Skip sequence since it has been processed before: %s' % pkl_file)
        return sequence_infos

    sequence_infos, sequence_points =\
        process_single_tfrecord_multiprocessing(
            sequence_file=sequence_file_tfrecord,
            has_label=has_label,
            pool=pool
        )

    for cnt in range(len(sequence_points)):
        infos = sequence_infos[cnt]
        lidar_path = os.path.join(cur_save_dir, ('%04d.npy' % cnt))
        infos['lidar_path'] = lidar_path
        
        np.save(lidar_path, sequence_points[cnt])

    # save the data infomation
    with open(pkl_file, 'wb') as f:
        pickle.dump(sequence_infos, f)
    
    return sequence_infos


def process_single_tfrecord_multiprocessing(sequence_file:str, has_label:bool=True, pool:multiprocessing.Pool=None):
    """ Process single sequence and return the preprocessing result
    
    Attributes:
        sequence_file: The path of tfrecord that to be processed
        has_label: True if tfrecord include grount-truth labels (trainval set), False otherwise (test set or new sequence)
        pool: The multiprocessing pool. leave it blank if you dont want to use multiprocessing.

    Returns:
        sequence_infos: A list of infos for each frame
        sequence_points: A list of points for each frame

    Raises:
        IOError: An error occurred when processing the tfrecord file
        FileNotFoundError: The tfrecord file to be processed is not exists
    """

    dataset = tf.data.TFRecordDataset(str(sequence_file), compression_type='')
    sequence_name = os.path.splitext(os.path.basename(sequence_file))[0]
    sequence_name = sequence_name.strip('segment-').strip('_with_camera_labels')
    
    # pre-load tfrecords
    frame_data_list = []
    sequence_infos = []
    for cnt, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_data_list.append((cnt, frame))

        info = {
            'time_stamp': frame.timestamp_micros,
            'sample_idx': cnt,
            'sequence_name': sequence_name,
            'pose': np.array(frame.pose.transform, dtype=np.float32).reshape(4, 4)
        }

        if has_label:
            annotations = generate_labels(frame)
            info['annos'] = annotations
            info['annos']['gt_boxes_global'] = box_utils.transform_boxes3d(info['annos']['gt_boxes_lidar'], info['pose'])
        sequence_infos.append(info)
    
    # process point clouds by multiprocessing
    if pool is not None:
        data_list = pool.map(save_data_worker, [frame for _, frame in frame_data_list])
    else:
        data_list = map(save_data_worker, [frame for _, frame in frame_data_list])
    sequence_points, num_points_of_each_lidar_list = zip(*data_list)
    
    for idx in range(len(sequence_infos)):
        info = sequence_infos[idx]
        info['num_points_of_each_lidar'] = num_points_of_each_lidar_list[idx]
        info['lidar_path'] = 'None'
        info['sequence_len'] = len(sequence_infos)

    return sequence_infos, sequence_points


def save_data_worker(frame):
    range_images, camera_projections, range_image_top_pose = \
        frame_utils.parse_range_image_and_camera_projection(frame)

    points, cp_points, points_in_NLZ_flag, points_intensity, points_elongation = \
        convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose)

    # 3d points in vehicle frame
    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

    num_points_of_each_lidar = [point.shape[0] for point in points]
    save_points = np.concatenate([
        points_all, points_intensity, points_elongation, points_in_NLZ_flag
    ], axis=-1).astype(np.float32)
    
    return save_points, num_points_of_each_lidar

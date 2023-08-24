# DetZero - Evaluation Tool


## Intro

- This is an offline tool for evaluating DetZero detection results. You can use the detection results of the DetZero pipeline to conduct the evaluation.

- You must have the detection result and the ground-truth annotations as the input and the evaluator will invoke the `waymo_open_dataset` tool to assess the performance.

- You can make the prediction without considering the order, to do so, you must include the name of sequence (`sequence_name`) and the index of the frame in the sequence (`frame_id`) in the generated detection result.

- In order to excute the evaluation, you will also need a ground-truth annotations. We have provided the scripts to generate these files, which can be stoted in the default path: `DetZero/data/waymo/`.


## Running
- a. generate ground truth pickle files


- b. evaluate the performance
```
python -u detzero_eval.py [-h]
                          [--det_result_path DET_RESULT_PATH]
                          [--gt_info_path GT_INFO_PATH]
                          [--evaluate_metrics EVALUATE_METRICS [EVALUATE_METRICS ...]]
                          [--class_name CLASS_NAME [CLASS_NAME ...]]
                          [--iou_type IOU_TYPE]
                          [--info_with_fakelidar]
                          [--distance_thresh DISTANCE_THRESH]

optional arguments:
  -h, --help        show this help message and exit
  
  --det_result_path DET_RESULT_PATH
                    path to the prediction result pickle file
  
  --gt_info_path GT_INFO_PATH
                    path to the generated ground-truth info pickle file
  
  --evaluate_metrics EVALUATE_METRICS [EVALUATE_METRICS ...]
                    metrics that used in evaluation, support multiple (object, range)
  
  --class_name CLASS_NAME [CLASS_NAME ...]
                    the class names of the detection, default=["Vehicle", "Pedestrian", "Cyclist"]
  
  --iou_type IOU_TYPE
                    the box matching type, support '3d' or 'bev'
  
  --info_with_fakelidar
  
  --distance_thresh DISTANCE_THRESH
                    the perception range for evaluation
  
  --tracking        store True to evaluate the tracking performance (currently only support L2 difficulty)
  
  --human_study     store True to evalute the selected sequences for human study
```
An example:
```shell
cd DetZero/evaluator &&
python -u detzero_eval.py \
  --det_result_path ../detection/output/centerpoint_1sweep/default/eval/epoch_30/val/result.pkl \
  --gt_info_path ../data/waymo/waymo_processed_val.pkl
  --evaluate_metrics object range
```
`--evaluate_metrics` indicates the desired evaluation that you want to conduct. Support `object` (by default), `range`, or, `object range`. More metrics means much elapsed time, `object` is fastest and adequate for most circumstance, `range` will output the performance splited by different ranges.


## Waymo Results Sumbmission
- You can use `waymo_submit.py` to generate a bin file for Waymo leaderboard submission.
- You can find more details about Waymo leaderboard sumbission with fowllowing [link](https://mmdetection3d.readthedocs.io/en/latest/advanced_guides/datasets/waymo.html).

For example:
```shell
cd DetZero/evaluator &&
python -u waymo_submit.py --pred_path <PRED_DATA_PATH> --gt_path <GT_INFO_PATH> --output_path <OUTPUT_DIR_PATH>
```

## Layout of prediction result pickle

The prediction pickle file includes a list of dicts. Each dict indicates a frame. Here is an layout example of a dict which contains N detected objects:
```
[
    {
        'sequence_name': a string like 'segment-10203656353524179475_7625_000_7645_000',
        'frame_id': an int that indicates the index of the frame in the sequence,
        'name': an numpy string array in [N],
        'score': an numpy float32 array in [N],
        'boxes_lidar': an numpy float32 array in [N, 7],
        'pose': an numpy float32 array in [4, 4]
    },
    {
        ...
    },
    ...
]
```

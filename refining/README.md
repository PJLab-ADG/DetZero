# DetZero - Refining Module


## Intro
- This is the Refining module of DetZero framework. It contains the full training procedure for `Geometry`, `Position`, `Confidence` Refining Models.

- Before training, we must prepare the object data (mentioned in Sec. 3.2 of paper, `step b` of following commands), which mainly includes the object-specific LiDAR points cropping, and ground-truth boxes matching (`matched` and `matched_tracklet` tag in our code).

- For the training process, the GRM and PRM should be trained firstly. Then, we will combine the inference results of these two models as the refined 3D boxes results, to generate IoU related labels for CRM（`step e, f` of following commands）.

- The final result will be the combination of the inference results of GRM, PRM and CRM (`step f` of following commands).


## Structure Tree of Refining Module
```
	DetZero
	└── refining
	    ├── detzero_refine
	    |   ├── datasets: different data processing and loading
	    |   ├── models 
	    |   |   ├── refining model wrappers
	    |   |   └── modules: transformer modules and related structures
	    |   └── utils: the exclusive utils used in refining module
	    └── tools
	        ├── cfgs: dataset and model config files
	        └── train and evaluate scripts
```


## Commands

- a. compile the module
```shell
cd DetZero/refining &&
python setup.py develop
```

- b. prepare object data (see [Daemon](../daemon/README.md) for more details)
```shell
cd DetZero/daemon &&
python prepare_object_data.py --track_data_path <PATH_TO_TRACKING_RESULT> --split <DATA_SPLIT>
```

- c. train GRM
```shell
cd DetZero/refining/tools &&
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml
```

- d. train PRM
```shell
cd DetZero/refining/tools &&
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml
```

- e. inference geometry and position results for training set
```shell
cd DetZero/refining/tools

python test.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml --ckpt <PATH_TO_GRM_CKPT> --extra_tag <YOUR_EXTRA_TAG> --save_to_file --set DATA_CONFIG.DATA_SPLIT.test train

python test.py --cfg_file cfgs/ref_model_cfgs/vehicle_grm_model.yaml --ckpt <PATH_TO_PRM_CKPT> --extra_tag <YOUR_EXTRA_TAG> --save_to_file --set DATA_CONFIG.DATA_SPLIT.test train
```

- f. generate IoU labels for CRM (see [Daemon](../daemon/README.md) for more details)
```shell
cd DetZero/daemon &&
python generate_iou_gt.py --class_name <CLASS_NAME> --geo_path <PATH_TO_GEOMETRY_RESULT> --pos_path <PATH_TO_POSITION_RESULT>
```

- g. train CRM
```shell
cd DetZero/refining/tools &&
python train.py --cfg_file cfgs/ref_model_cfgs/vehicle_crm_model.yaml
```

- h. inference the results for validation set with the best checkpoints of these 3 refining models respectively
```shell
cd DetZero/refining/tools &&
python test.py --cfg_file <MODEL_CONFIG> --ckpt <PATH_TO_CKPT> --extra_tag <YOUR_EXTRA_TAG> --save_to_file --set DATA_CONFIG.DATA_SPLIT.test val
```

- i. combine the results of these 3 refining models together as the final output result (see [Daemon](../daemon/README.md) for more details)
```shell
cd DetZero/daemon &&
python combine_output.py --split val --combine_drop_path <PATH_TO_DROP_DATA> --combine_conf_res --track_save --frame_save
```


## Note
- We suggest you to use A100 for training, or other devices with more memories. Compared to the time consuming of opening files frequently, we would like to load all the needed infos in the memory and speed up the training process.

- We give the default value of `batch size` under 1 gpu setting, please specify the value based on your own machine setting with `--set OPTIMIZATION.BATCH_SIZE_PER_GPU default_value / gpu_num`. If there are no A100s, we suggest to use multi-gpu training command to satisfy the requriements of `batch size`.

- For better data management, we suggest you to move the inference results of the best models of GRM, PRM , and CRM, to the folder of `DetZero/data/waymo/refining/results`.

- The performance of CRM is non-trival to evaluate straightforwad, we combine the results of CRM with GRM and PRM together to evaluate waymo mAP.


# DetZero - Tracking Module


## Intro
- This is the offline tracking module of DetZero framework. It is designed not only to associate detected boxes as object tracks, but also to prepare ground-truth related information for downstream refining modules.

- Before running tracking module, we must prepare the detected bounding boxes with the format of pickle (the name should be indicated in the dataset config file `DetZero/trakcing/tools/tk_dataset_cfgs/waymo_dataset.yaml`), and place it at `DetZero/data`.

- The ground-truth files for train and validation split are also needed, please note that the `generate_gt_info.py` is executed and the corresponding gt_infos pickle files are generated at `DetZero/data/gt_infos/`.


## Structure Tree of Tracking Module
```
	DetZero
	└── tracking
	    ├── detzero_track
	    |   ├── datasets: processor and dataloader of dataset
	    |   ├── models: implemention of tracking methods  
	    |   └── utils: the exclusive utils used in tracking module
	    └── tools
	        ├── cfgs: dataset and model config files
	        ├── eval: evaluation tool
	        └── scripts: sh scripts
``` 


### Prerequisite:
- virtual environment & required packages (see details in [INSTALL.md](../docs/INSTALL.md))


### Running
- a. compile the tracking module
```shell
cd DetZero/tracking &&
python setup.py develop
```

- b. run tracking model in target assign mode (use gt annotations to assign labels for each object track)
```shell
cd DetZero/tracking/tools &&
python -u run_track.py --cfg_file cfgs/tk_model_cfgs/waymo_detzero_track.yaml --data_path <PATH_TO_DETECTION_RESULT> --workers <WORKER_NUM> --split <train or val>
```

- c. run tracking model for infering test set
```shell
cd DetZero/tracking/tools &&
python -u run_track.py --cfg_file cfgs/tk_model_cfgs/waymo_detzero_track.yaml --data_path <PATH_TO_DETECTION_RESULT> --workers <WORKER_NUM> --split test
```

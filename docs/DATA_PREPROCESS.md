# Data Preprocess

## Dataset Preparation
Currently we only provide the processing of Waymo dataset. 

- please place the soft link of the data folder based on the following structure:
	```
	detection
	├── data
	│   ├── waymo
	│   │   │── ImageSets
	│   │   │── raw_data
	│   │   │   │── segment-xxxxxxxx.tfrecord
	│   │   │   │── ....
	├── detzero_det
	├── tools
	```

- process waymo infos:
  ```shell
  cd detection
  python -m detzero_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml --func create_waymo_infos
  ```

- generate database for gt-sampling
  ```
  cd detection
  python -m detzero_det.datasets.waymo.waymo_preprocess --cfg_file tools/cfgs/det_dataset_cfgs/waymo_one_sweep.yaml --func create_waymo_database
  ```

### NOTE
We have provided a flexible data info structure and processing logic to satisfy single-frame or multi-frame (e.g., 2, 3, 5, ...) point clouds loading without further repeated pre-processings.
```python
    def get_sweep_idxs(current_info, sweep_count=[0, 0], current_idx=0):

        assert type(sweep_count) is list and len(sweep_count) == 2,\
            "Please give the upper and lower range of frames you want to process!"

        current_sample_idx = current_info["sample_idx"]
        current_seq_len = current_info["sequence_len"]

        target_sweep_list = np.array(list(range(sweep_count[0], sweep_count[1]+1)))
        target_sample_list = current_sample_idx + target_sweep_list
        # set the low and high thresh to extract multi frames in current sequence
        target_sample_list = [i if i >= 0 else 0 for i in target_sample_list]
        target_sample_list = [i if i < current_seq_len else current_seq_len-1 for i in target_sample_list]
        # get the index of target frames in the waymo info list
        target_idx_res = np.array(target_sample_list) - current_sample_idx
        target_idx_list = current_idx + target_idx_res

        return target_idx_list
```


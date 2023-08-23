# Installation

## Prerequisites
All the codes are tested in the environment:
- Linux (tested on Ubuntu 16.04/18.04/20.04)
- Python 3.8 (waymo-open-dataset 2-5-0 required)
- PyTorch 1.10 or higher
- CUDA 11.0 or higher
- GCC 5.4+
- [spconv v2.x](https://github.com/traveller59/spconv)


## Recomended Steps
**a. Create a conda virtual environment.**
```shell
  conda create --name detzero python=3.8
  conda activate detzero
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
  pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
```

Note: Make sure that your compilation CUDA version and runtime CUDA version match.

**c. Install cmake.**
```shell
  conda install cmake
```

**d. Install sparse conv.**
```shell
  pip install spconv-cu111
```

**e. Install pytorch scatter (for DynamicVFE).**
We suggest to follow the instructions of [torch_scatter](https://github.com/rusty1s/pytorch_scatter) to install the package based on your own environment [version](https://data.pyg.org/whl/).


**f. Install Waymo evaluation module.**
```shell
  pip install waymo-open-dataset-tf-2-5-0
```

**g. Install other required dependent libraries.**
```shell
  cd DetZero && pip install -r requirements.txt
```

**h. Compile other libraries.**
```shell
  cd DetZero/utils && python setup.py develop
```

**i. Compile the libraries of specific algorithm modules.**
```shell
  cd DetZero/detection && python setup.py develop
```
```shell
  cd DetZero/tracking && python setup.py develop
```
```shell
  cd DetZero/refining && python setup.py develop
```

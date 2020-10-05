# ISVC2020 Fundus2Angio
This code is part of the supplementary materials for the ISVC 2020 for our paper Fundus2Angio: A Conditional GAN Architecture for Generating Fluorescein Angiography Images from Retinal Fundus Photography . The paper has since been accpeted to ISVC 2020 and will be preseneted in October 2020.

### Arxiv Pre-print
```
https://arxiv.org/abs/2005.05267
```
### IEEE Xplore Digital Library
```
https://ieeexplore.ieee.org/document/9190742
```

# Citation 
```
@INPROCEEDINGS{9190742,

  author={S. A. {Kamran} and A. {Tavakkoli} and S. L. {Zuckerbrod}},

  booktitle={2020 IEEE International Conference on Image Processing (ICIP)}, 

  title={Improving Robustness Using Joint Attention Network for Detecting Retinal Degeneration From Optical Coherence Tomography Images}, 

  year={2020},

  volume={},

  number={},

  pages={2476-2480},}
```

## Pre-requisite
- Ubuntu 18.04 / Windows 7 or later
- NVIDIA Graphics card

## Installation Instruction for Ubuntu
- Download and Install [Nvidia Drivers](https://www.nvidia.com/Download/driverResults.aspx/142567/en-us)
- Download and Install via Runfile [Nvidia Cuda Toolkit 10.0](https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=runfilelocal)
- Download and Install [Nvidia CuDNN 7.6.5 or later](https://developer.nvidia.com/rdp/cudnn-archive)
- Install Pip3 and Python3 enviornment
```
sudo apt-get install pip3 python3-dev
```
- Install Tensorflow-Gpu version-2.0.0 and Keras version-2.3.1
```
sudo pip3 install tensorflow-gpu==2.0.3
sudo pip3 install keras==2.3.1
```
- Install packages from requirements.txt
```
sudo pip3 -r requirements.txt
```

### Dataset download link for Hajeb et al.
```
https://sites.google.com/site/hosseinrabbanikhorasgani/datasets-1/fundus-fluorescein-angiogram-photographs--colour-fundus-images-of-diabetic-patients
```

# EBSR: Feature Enhanced Burst Super-Resolution With Deformable Alignment 
## Dependencies
- OS: Ubuntu 18.04
- Python: Python 3.7
- nvidia :
   - cuda: 10.1
   - cudnn: 7.6.1
- Other reference requirements

## Quick Start
1.Create a conda virtual environment and activate it
```python3
conda create -n pytorch_1.6 python=3.7
source activate pytorch_1.6
```
2.Install PyTorch and torchvision following the official instructions
```python3
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch
```
3.Install build requirements
```python3
pip3 install -r requirements.txt
```
4.Install apex to use DistributedDataParallel following the [Nvidia apex](https://github.com/NVIDIA/apex)
```python3
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
```
5.Install DCN
```python3
cd DCNv2-pytorch_1.6
python3 setup.py build develop # build
python3 test.py # run examples and check
```
## Training
```python3
# Modify the root path of training dataset and model etc.
# The number of GPUs should be more than 1
python3 main.py --n_GPUs 2 --lr 0.0004 --root /data/ntire/ --model EBSR
```
## Test
```python3
# Modify the path of test dataset and the path of the trained model
python3 test.py
```
## Citations
If EBSR helps your research or work, please consider citing EBSR.
The following is a BibTeX reference. 

```
@InProceedings{Luo_2021_CVPR,
    author    = {Luo, Ziwei and Yu, Lei and Mo, Xuan and Li, Youwei and Jia, Lanpeng and Fan, Haoqiang and Sun, Jian and Liu, Shuaicheng},
    title     = {EBSR: Feature Enhanced Burst Super-Resolution With Deformable Alignment},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2021},
    pages     = {471-478}
}
```

## Contact
email: [algo_lzw@yahoo.com, yl_yjsy@163.com]
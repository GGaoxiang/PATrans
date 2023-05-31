## Installation

### Requirements:
- PyTorch 1.3
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 9.2


### installation on Linux

```bash
# create a conda virtual environment
conda create --name PATrans -y python=3.7
source activate PATrans

# install the right pip and dependencies
conda install ipython pip
pip install ninja yacs cython matplotlib tqdm opencv-python scipy

# PyTorch installation with CUDA 10.0
conda install pytorch=1.3.0 torchvision cudatoolkit=10.0 -c pytorch

# install pycocotools
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

<!-- # install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext -->

# install PATrans
git clone https://github.com/GGaoxiang/PATrans.git

# compile
python setup.py build develop 
# if you are using slurm cluster, you can run the following command. (Need to define the <partition>)
```bash
bash compile.sh
```

```bash
pip install 'pillow<7.0.0'
pip install tensorboardX mmcv
```

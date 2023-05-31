# Example for environment (Replace the following path with your own)
export LD_LIBRARY_PATH=/mnt/lustre/gaoxiang.vendor/anaconda3/lib:$LD_LIBRARY_PATH
export PATH=/mnt/lustre/gaoxiang.vendor/anaconda3/bin:$PATH
#python3.7, pytorch1.3, cuda10.0, gcc5.4
export LD_LIBRARY_PATH=/mnt/cache/share/cuda-10.0/lib64:$LD_LIBRARY_PATH
export PATH=/mnt/cache/share/cuda-10.0/bin:$PATH
export LD_LIBRARY_PATH=/mnt/cache/share/gcc/gcc-5.4/lib64:$LD_LIBRARY_PATH
export PATH=/mnt/cache/share/gcc/gcc-5.4/bin:$PATH
export CUDA_HOME=/mnt/cache/share/cuda-10.0
#conda activate /mnt/lustre/gaoxiang.vendor/anaconda3/envs/PATrans 
#conda deactivate
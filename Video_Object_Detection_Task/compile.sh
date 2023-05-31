worker=pat_uranus #<partition> # Need to define the <partition>
work_num=1
srun -p $worker -w SH-IDC1-10-198-8-46 -n1 --gres=gpu:$work_num \
/mnt/lustre/gaoxiang.vendor/anaconda3/envs/STFT/bin/python setup.py build develop
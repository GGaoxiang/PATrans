import os
import random
import torch
import torch.distributed as dist
import re
import numpy as np
from patrans_core.utils import gpu_indices, ompi_size, ompi_rank, get_master_ip


def init_dist(launcher, args, backend='nccl'):
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, args)
    elif launcher == 'mpi':
        _init_dist_mpi(backend, args)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, args)
    else:
        raise ValueError('Invalid launcher type: {}'.format(launcher))


def _init_dist_pytorch(backend, args):
    os.environ['MASTER_PORT'] = args.master_port
    # os.environ['MASTER_PORT'] = str(int(args.master_port) + int(random.randint(0,50)))
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        backend=backend, init_method="env://"
    )


def _init_dist_mpi(backend, args):
    gpus = list(gpu_indices())
    gpu_num = len(gpus)
    world_size = ompi_size()
    rank = ompi_rank()
    dist_url = 'tcp://' + get_master_ip() + ':23456'
    torch.cuda.set_device(int(gpus[0]))  # Set current GPU to the first
    dist.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        group_name='mtorch')
    print(
        "World Size is {}, Backend is {}, Init Method is {}, rank is {}, gpu num is{}"
        .format(world_size, backend, dist_url, ompi_rank(), gpu_num))

def _init_dist_slurm(backend, args):
    try:
        print("init dist!!")
        # get env info
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = int(os.environ['SLURM_NTASKS'])
        args.local_rank = int(os.environ['SLURM_LOCALID'])
        node_list = str(os.environ['SLURM_NODELIST'])
        print('rank:{}, world_size:{}, local_rank:{}, node_list:{}'.format(args.rank, args.world_size, args.local_rank, node_list))
        # set ip address
        node_parts = re.findall('[0-9]+', node_list)
        host_ip = '{}.{}.{}.{}'.format(node_parts[1],node_parts[2],node_parts[3],node_parts[4])
        # port should not be used
        port = args.master_port
        # initialize tcp method
        init_method = 'tcp://{}:{}'.format(host_ip, port)  
        # initialize progress
        try:
            dist.init_process_group("nccl", init_method=init_method, world_size=args.world_size, rank=args.rank)
        except:
            dist.init_process_group("nccl")
        # set device for each node
        torch.cuda.set_device(args.local_rank)
        print('ip: {}'.format(host_ip))

    except:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.local_rank
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        args.local_rank = torch.distributed.get_rank()
        args.rank = torch.distributed.get_rank()
        args.world_size = torch.distributed.get_world_size()
        print('run on local', 'rank:', args.rank, 'world_size:', args.world_size)


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    # if cuda_deterministic:  # slower, more reproducible
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # else:  # faster, less reproducible
    #     torch.backends.cudnn.deterministic = False
    #     torch.backends.cudnn.benchmark = True
    
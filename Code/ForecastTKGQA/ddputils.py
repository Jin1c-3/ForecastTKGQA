import numpy as np
import torch
import random
import os
from torch import distributed as dist
from contextlib import contextmanager


def execute_once(func):
    def wrapper(*args, **kwargs):
        if should_execute():
            return func(*args, **kwargs)

    return wrapper


def should_execute():
    return get_rank() == 0


def init_distributed_env(args):
    dist.init_process_group("gloo", init_method="env://")
    synchronize()


def set_rand_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)  # Disable hash randomization
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_rank():
    if dist.is_initialized():
        return dist.get_rank()
    if "RANK" in os.environ:
        return int(os.environ["RANK"])
    return 0


def get_device():
    devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if devices is None:
        raise Exception("CUDA_VISIBLE_DEVICES is not set")
    return torch.device("cuda", devices.split(",")[get_rank()])


def get_world_size():
    if dist.is_initialized():
        return dist.get_world_size()
    if "WORLD_SIZE" in os.environ:
        return int(os.environ["WORLD_SIZE"])
    return 1


def synchronize():
    if get_world_size() > 1:
        dist.barrier()


def reduce_scalars(*values, average=True):
    if get_world_size() < 2:  # 单GPU的情况
        return values

    reduced_values = []
    with torch.no_grad():
        for value in values:
            dist.all_reduce(value)
            if average:
                value /= get_world_size()
            reduced_values.append(value)

    if len(reduced_values) == 1:
        return reduced_values[0]
    return tuple(reduced_values)


def gather_tensors(*tensors):
    if get_world_size() < 2:  # 单GPU的情况
        return tensors

    reduced_tensors = []
    with torch.no_grad():
        for tensor in tensors:
            tensor_list = [torch.empty_like(tensor) for _ in range(get_world_size())]
            dist.all_gather(tensor_list, tensor)
            reduced_tensors.append(torch.cat(tensor_list, dim=0))

    if len(reduced_tensors) == 1:
        return reduced_tensors[0]
    return tuple(reduced_tensors)


def move_to_cuda(*values):
    return tuple(
        (
            torch.from_numpy(v).to(get_rank())
            if isinstance(v, np.ndarray)
            else v.cuda(get_rank())
        )
        for v in values
    )

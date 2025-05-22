import torch
import numpy as np

def frob_diff(pred, true):
    return torch.sqrt(torch.sum(torch.square(pred - true)))


def squared_diff(pred, true):
    return torch.sum(torch.square(pred - true))


def rmse(pred, true):
    return torch.sqrt(torch.mean(torch.square(pred - true)))


def running_frob(pred, true):
    num = torch.sqrt(torch.sum(torch.square(pred - true)))
    den = torch.sqrt(torch.sum(torch.square(true))) + 1e-8
    return num / den


def tensor_frob(tensor):
    sum_ = np.array([(t ** 2).sum() for t in tensor])
    return torch.sum(torch.from_numpy(sum_)).sqrt()


def tensor_squared(tensor):
    sum_ = np.array([(t ** 2).sum() for t in tensor])
    return torch.sum(torch.from_numpy(sum_))
import torch


def MSELoss(pred, gt):
    return torch.mean((pred - gt) ** 2)

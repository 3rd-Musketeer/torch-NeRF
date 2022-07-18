import numpy as np
import torch


def MSELoss(pred, gt):
    return torch.mean((pred - gt) ** 2)


def PSNR(pred, gt):
    return 10 * torch.log10(1.0 / torch.mean((pred - gt) ** 2))

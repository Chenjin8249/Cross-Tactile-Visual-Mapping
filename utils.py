# -*- coding: utf-8 -*-
import os
import sys
import math
import datetime
import struct
import time
import argparse
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from PIL import Image

from pytorch_msssim import ms_ssim
import torch.nn.functional as F
import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor


from pytorch_msssim import ms_ssim
from typing import Tuple, Union
import numpy as np
import torch.optim as optim


def parse_args():
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-e",
        "--epochs",
        default=800,
        type=int,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--metrics",
        default='ccl',  # mse / ms-ssim
        type=str,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--checkpoint",
        default='',
        type=str,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--train_dataset",
        default='./data/TrainDataFile',
        type=str,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "--test_dataset",
        default='./data/TestDataFile',
        type=str,
        help="Number of epochs (default: %(default)s)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        default=5e-4,
        type=float,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument(
        "-n",
        "--num-workers",
        type=int,
        default=4,
        help="Dataloaders threads (default: %(default)s)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: %(default)s)")  # 8
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1,
        help="Test batch size (default: %(default)s)",
    )

    parser.add_argument("--seed", type=int, default=1, help="Set random seed for reproducibility")
    parser.add_argument(
        "--clip_max_norm",
        default=1.0,
        type=float,
        help="gradient clipping max norm (default: %(default)s",
    )
    args = parser.parse_args()
    return args


def save_checkpoint(state, is_best, filename):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('checkpoint', 'checkpoint_best_loss'))


def compute_joint(view1, view2):
    """Compute the joint probability matrix P"""

    bn, k = view1.size()
    assert (view2.size(0) == bn and view2.size(1) == k)

    view1 = F.softmax(view1, dim=1)
    view2 = F.softmax(view2, dim=1)
    p_i_j = view1.unsqueeze(2) * view2.unsqueeze(1)
    p_i_j = p_i_j.sum(dim=0)
    p_i_j = (p_i_j + p_i_j.t()) / 2.  # symmetrise
    p_i_j = p_i_j.clamp_min(torch.finfo(p_i_j.dtype).eps)
    p_i_j = p_i_j / p_i_j.sum()  # normalise

    return p_i_j


# def crossview_contrastive_Loss(view1, view2, lamb=9, EPS=sys.float_info.epsilon):
#     """Contrastive loss for maximizng the consistency"""
#     n, k = view1.size()
#     p_i_j = compute_joint(view1, view2)
#     assert (p_i_j.size() == (k, k))
#
#     p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
#     p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)
#
#     #     Works with pytorch <= 1.2
#     #     p_i_j[(p_i_j < EPS).data] = EPS
#     #     p_j[(p_j < EPS).data] = EPS
#     #     p_i[(p_i < EPS).data] = EPS
#
#     # Works with pytorch > 1.2
#     p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
#     p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
#     p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)
#
#
#     loss = - 3 * p_i_j * torch.log(p_i_j) + 2 * p_i * torch.log(p_i) + 2 * p_j * torch.log(p_j)
#
#     loss = loss.sum()/(k*k)
#
#     return loss
def crossview_contrastive_Loss(view1, view2, lamb=9.0, EPS=sys.float_info.epsilon):
    """Contrastive loss for maximizng the consistency"""
    _, k = view1.size()
    p_i_j = compute_joint(view1, view2)
    assert (p_i_j.size() == (k, k))

    p_i = p_i_j.sum(dim=1).view(k, 1).expand(k, k)
    p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)

    #     Works with pytorch <= 1.2
    #     p_i_j[(p_i_j < EPS).data] = EPS
    #     p_j[(p_j < EPS).data] = EPS
    #     p_i[(p_i < EPS).data] = EPS

    # Works with pytorch > 1.2
    p_i_j = torch.where(p_i_j < EPS, torch.tensor([EPS], device=p_i_j.device), p_i_j)
    p_j = torch.where(p_j < EPS, torch.tensor([EPS], device=p_j.device), p_j)
    p_i = torch.where(p_i < EPS, torch.tensor([EPS], device=p_i.device), p_i)

    loss = - p_i_j * (torch.log(p_i_j) \
                      - (lamb + 1) * torch.log(p_j) \
                      - (lamb + 1) * torch.log(p_i))

    loss = loss.sum()/(k*k)

    return loss

def compute_metrics(
        a: Union[np.array, Image.Image],
        b: Union[np.array, Image.Image],
        max_val: float = 255.0,
) -> Tuple[float, float]:
    """Returns PSNR and MS-SSIM between images `a` and `b`. """
    if isinstance(a, Image.Image):
        a = np.asarray(a)
    if isinstance(b, Image.Image):
        b = np.asarray(b)

    a = torch.from_numpy(a.copy()).float().unsqueeze(0).unsqueeze(0)
    if a.size(3) == 3:
        a = a.permute(0, 3, 1, 2)
    b = torch.from_numpy(b.copy()).float().unsqueeze(0).unsqueeze(0)
    if b.size(3) == 3:
        b = b.permute(0, 3, 1, 2)

    mse = torch.mean((a - b) ** 2).item()
    p = 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


class Loss(nn.Module):

    def __init__(self, metrics='ccl'):
        super().__init__()
        self.mse = nn.MSELoss()
        self.metrics = metrics

    def forward(self, SourceLatent, TargetLatent, SourcePredict, TargetPredict, source, target):
        out = {}

        out["mse_loss_T"] = self.mse(TargetPredict, target)
        out["mse_loss_S"] = self.mse(SourcePredict, source)
        out["ms_ssim_loss"] = 1 - ms_ssim(TargetPredict, target, data_range=1.0)
        out["crossview_contrastive_Loss"] = crossview_contrastive_Loss(SourceLatent.view(SourceLatent.shape[0], -1), TargetLatent.view(TargetLatent.shape[0], -1))
        out["loss"] = 0.2 * out["crossview_contrastive_Loss"] + 0.4 * out["mse_loss_T"] + 0.4 * out["mse_loss_S"]
        # out["loss"] =out["mse_loss_S"]
        # out["loss"] =  0.5 * out["ms_ssim_loss"]
        # if self.metrics == 'mse':
        #     out["mse_loss"] = self.mse(TargetPredict, target)
        #     out["ms_ssim_loss"] = None
        #     out["loss"] = out["mse_loss"]
        # elif self.metrics == 'ms-ssim':
        #     out["mse_loss"] = self.mse(TargetPredict, target)
        #     out["ms_ssim_loss"] = 1 - ms_ssim(TargetPredict, target, data_range=1.0)
        #     out["loss"] = out["ms_ssim_loss"]
        # elif self.metrics == 'cl':
        #     out["mse_loss"] = self.mse(SourcePredict, target)
        #     out["ms_ssim_loss"] = 1 - ms_ssim(TargetPredict, target, data_range=1.0)
        #     out["crossview_contrastive_Loss"] = crossview_contrastive_Loss(SourceLatent, TargetLatent)
        #     out["loss"] = 0.5*out["crossview_contrastive_Loss"] + 0.5*out["ms_ssim_loss"]
        return out


def configure_optimizers(net, args):
    optimizer = optim.Adam(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.999))
    # optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.Adamax(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.9999))
    # optimizer = torch.optim.AdamW(net.parameters(), lr=args.learning_rate, betas=(0.9, 0.9999))
    return optimizer


class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def get_timestamp():
    return datetime.datetime.now().strftime('%y%m%d-%H%M%S')


def cal_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return transforms.ToTensor()(img)


def rename_key(key):
    # Deal with modules trained with DataParallel
    if key.startswith("module."):
        key = key[7:]

    # ResidualBlockWithStride: 'downsample' -> 'skip'
    if ".downsample." in key:
        return key.replace("downsample", "skip")

    # EntropyBottleneck: nn.ParameterList to nn.Parameters
    if key.startswith("entropy_bottleneck."):
        if key.startswith("entropy_bottleneck._biases."):
            return f"entropy_bottleneck._bias{key[-1]}"

        if key.startswith("entropy_bottleneck._matrices."):
            return f"entropy_bottleneck._matrix{key[-1]}"

        if key.startswith("entropy_bottleneck._factors."):
            return f"entropy_bottleneck._factor{key[-1]}"

    return key


def load_pretrained(state_dict):
    state_dict = {rename_key(k): v for k, v in state_dict.items()}
    return state_dict


def compute_psnr(a, b):
    mse = torch.mean((a - b) ** 2).item()
    return -10 * math.log10(mse)


def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_bpp(out_net):
    size = out_net['x_hat'].size()
    num_pixels = size[0] * size[2] * size[3]
    return sum(torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
               for likelihoods in out_net['likelihoods'].values()).item()


def Average(lst):
    return sum(lst) / len(lst)


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.detach().cpu().clamp(0, 1).squeeze())


if __name__ == "__main__":
    pass

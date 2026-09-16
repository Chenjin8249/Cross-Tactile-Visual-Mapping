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
from losses import Loss, compute_joint, crossview_contrastive_Loss


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
        default='ccl', choices=['ccl'],
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
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size (default: %(default)s)")  # 8
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
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda or cuda:0")
    args = parser.parse_args()
    return args


def save_checkpoint(state, is_best, filename):
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename.replace('checkpoint', 'checkpoint_best_loss'))


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
    p = float("inf") if mse == 0 else 20 * np.log10(max_val) - 10 * np.log10(mse)
    m = ms_ssim(a, b, data_range=max_val).item()
    return p, m


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

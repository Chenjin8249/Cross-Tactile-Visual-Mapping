import argparse
import random
import json
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import Model
from datasets import Dataset
from utils import load_pretrained, AverageMeter
import glob
from sklearn.model_selection import StratifiedKFold

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def test():
    device = "cuda"
    net = Model()
    net = net.to(device)
    print(f'[*] Total Parameters = {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    log_name = f'Stage1_20230512_233404'
    restore_path = f'./logs/{log_name}/best_val_acc.pth'
    checkpoint = torch.load(restore_path, map_location='cpu')
    print(f"INFO Load Pretrained Model From Epoch {checkpoint['epoch']}...")
    print(f"INFO Pretrained Model Val-Top1 {checkpoint['top1']}%")
    net.load_state_dict(checkpoint["state_dict"])

    train_dataset = FeatureDataset('./data/train/train_feature', './data/train.txt')
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
    )
    val_dataset = FeatureDataset('./data/train/train_feature', './data/val.txt')
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=False,
    )

    net.eval()

    train_top1 = AverageMeter()
    train_top2 = AverageMeter()
    train_top3 = AverageMeter()

    test_top1 = AverageMeter()
    test_top2 = AverageMeter()
    test_top3 = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(train_dataloader):
            features = d[0].to(device)  # [batch, num]
            label = d[1].to(device)
            out_net = net(features)

            acc = accuracy(out_net, label, [1, 2, 3])
            train_top1.update(acc[0].detach().item())
            train_top2.update(acc[1].detach().item())
            train_top3.update(acc[2].detach().item())

        for i, d in enumerate(val_dataloader):
            features = d[0].to(device)  # [batch, num]
            label = d[1].to(device)
            out_net = net(features)

            acc = accuracy(out_net, label, [1, 2, 3])
            test_top1.update(acc[0].detach().item())
            test_top2.update(acc[1].detach().item())
            test_top3.update(acc[2].detach().item())
    print(f"Train {restore_path} [Top1={train_top1.avg:.3f}%, Top2={train_top2.avg:.3f}%, Top3={train_top3.avg:.3f}%]")
    print(f"Test {restore_path} [Top1={test_top1.avg:.3f}%, Top2={test_top2.avg:.3f}%, Top3={test_top3.avg:.3f}%]")

    print(f"Start Testing")
    test_result_dict = {}
    with torch.no_grad():
        feature_path = './data/test_A/test_A_feature'
        names = sorted(os.listdir(feature_path))

        for i, n in enumerate(names):
            path = os.path.join(os.path.join(feature_path), n)
            fea = np.load(path).reshape((250, 64, 32))
            fea = torch.from_numpy(fea).unsqueeze(0).to(device)
            out_net = torch.squeeze(net(fea)).cpu()
            predict_cla = torch.argmax(out_net).detach().item()
            test_result_dict[n] = predict_cla
        result = str(test_result_dict).replace("'","\"")
        print(result)
        with open(f'./logs/{log_name}/test.txt', 'w') as ft:
            ft.write(f'{result}')

    return 0


if __name__ == "__main__":
    test()

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import os
from glob import glob
from torchvision import transforms
from torch.utils.data.dataset import Dataset
import torch

import math
import torch.nn as nn


def read_txt(txt_path):
    # Preserve original mean padding, but ensure odd deficits also reach 1601.
    with open(txt_path, encoding="utf-8-sig") as stream:
        values = [float(line.strip()) for line in stream if line.strip()]
    ret = np.asarray(values, dtype=np.float32)
    if not ret.size or not np.isfinite(ret).all():
        raise ValueError(f"Empty or non-finite tactile data: {txt_path}")
    deficit = 1601 - ret.size
    if deficit < 0:
        raise ValueError(f"{txt_path}: {ret.size} values; expected <=1601. Supply the original preprocessing; no silent truncation.")
    return np.pad(ret, (deficit // 2, deficit - deficit // 2), mode="mean")


def label_fun(label_path):
    print(label_path)
    # Y_list = []
    # for labelname in label_path:
    label1 = label_path.split('_')[0]
    label2 = label1.split('\\')[2]
    label3 = label2[:-1] if label2[-1].isdigit() else label2
    # Y_list.append(label3)

    return label3

def read_npy_file(filename):
    data = np.load(filename)
    return data.astype(np.float32)

class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")

        self.source_files_paths = sorted(glob(os.path.join(self.data_dir, 'Source', "*.txt")))
        self.target_files_paths = sorted(glob(os.path.join(self.data_dir, 'Target', "*.jpg")))
        # self.label_files_paths = sorted(glob(os.path.join(self.data_dir, 'Target', "*.jpg")))

        assert len(self.source_files_paths) == len(self.target_files_paths), \
            f'{len(self.source_files_paths)} != {len(self.target_files_paths)}'

        if not self.source_files_paths:
            raise ValueError(f"No paired samples in {data_dir}")
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((256, 256))
             ])

    def __getitem__(self, item):
        source_file = self.source_files_paths[item]
        target_file = self.target_files_paths[item]
        # label_file = self.label_files_paths[item]
        # print(source_file, target_file)

        source = read_txt(source_file)
        # source = read_npy_file(source_file)
        with Image.open(target_file) as image:
            target = image.convert("L")
        # label = label_fun(label_file)
        # print(target.size, source.shape,label)

        return source, self.transform(target)
        # return self.transform(source), self.transform(target)

    def __len__(self):
        return len(self.target_files_paths)


def get_loader(train_data_dir, test_data_dir, batch_size):
    train_dataset = Datasets(train_data_dir)
    test_dataset = Datasets(test_data_dir)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)
    return train_loader, test_loader


if __name__ == "__main__":
    # train_data = Datasets('./data/TrainDataFile')   (320, 480) (1601,)
    # val_data = Datasets('./data/TestDataFile')  (320, 480) (1581,)
    # d0 = val_data[0]
    # print(len(val_data))

    train_loader, test_loader = get_loader('./data/TrainDataFile', './data/TestDataFile', 16)
    for batch in test_loader:
        source, target = batch
        print(source.shape, target.shape)

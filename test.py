"""Evaluate checkpoints produced by the corrected training entry point."""
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from datasets import Datasets
from model import Model
from losses import Loss
from train import test_epoch


class NoSummary:
    def add_scalar(self, *args, **kwargs):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--test_dataset', required=True)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--batch-size', type=int, default=1)
    args = parser.parse_args()
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    if checkpoint.get('revision') != 'unet_bidirectional_v2':
        raise ValueError('Use a checkpoint trained with this corrected package')
    model = Model()
    model.load_state_dict(checkpoint['state_dict'])
    del checkpoint
    model.to(device)
    loader = DataLoader(Datasets(args.test_dataset), batch_size=args.batch_size)
    logging.basicConfig(level=logging.INFO)
    # Epoch 0 evaluates without writing the 20-epoch preview outputs.
    test_epoch(0, loader, model, Loss(), 'evaluation', logging.getLogger('eval'), NoSummary())


if __name__ == '__main__':
    main()

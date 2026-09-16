import os
# Select devices with --device; preserve the caller's CUDA visibility.
from glob import glob
import shutil
import math
import random
import json
import sys
import datetime
from tqdm import tqdm
import numpy as np
from pytorch_msssim import ms_ssim

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# import logging
# from utils import setup_logger, AverageMeter, CustomDataParallel, parse_args, torch2img, \
#     compute_metrics, configure_optimizers, Loss, save_checkpoint
from model import Model
from datasets import Datasets
# from CompleterModel import *
from utils import *


def train_one_epoch(model, criterion, train_dataloader, optimizer, epoch, clip_max_norm,
                    logger_train, tb_logger, current_step, args):
    model.train()
    device = next(model.parameters()).device

    train_loss = AverageMeter()
    train_mse_T = AverageMeter()
    train_mse_S = AverageMeter()
    train_psrn = AverageMeter()
    train_ms_ssim = AverageMeter()
    train_ccl = AverageMeter()

    for i, d in enumerate(train_dataloader):
        source = d[0].to(device).float()
        target = d[1].to(device)
        # label = d[2].to(device)
        # print(label)

        optimizer.zero_grad()

        SourceLatent, TargetLatent, SourcePredict, TargetPredict = model(source, target)

        # predict = model(source)

        out_criterion = criterion(SourceLatent, TargetLatent, SourcePredict, TargetPredict, source, target)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        psrn = -10 * np.log10(max(out_criterion["mse_loss_T"].item(), 1e-12))
        msssim = 1 - out_criterion["ms_ssim_loss"]
        # ccl = crossview_contrastive_Loss(SourceLatent, TargetLatent)

        train_loss.update(out_criterion["loss"].item(), target.size(0))
        train_psrn.update(psrn, target.size(0))
        train_ms_ssim.update(msssim.detach().item(), target.size(0))
        train_mse_T.update(out_criterion["mse_loss_T"].detach().item(), target.size(0))
        train_mse_S.update(out_criterion["mse_loss_S"].detach().item(), target.size(0))
        train_ccl.update(out_criterion["crossview_contrastive_Loss"].detach().item(), target.size(0))

        current_step += 1
        if current_step % 100 == 0:
            tb_logger.add_scalar('[train]: loss', out_criterion["loss"].item(), current_step)
            # if out_criterion["mse_loss"] is not None:
            #     tb_logger.add_scalar('{}'.format('[train]: mse_loss'), out_criterion["mse_loss"].item(), current_step)
            if out_criterion["ms_ssim_loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: ms_ssim_loss'), out_criterion["ms_ssim_loss"].item(),
                                     current_step)
            if out_criterion["mse_loss_T"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss_T'), out_criterion["mse_loss_T"].item(),
                                     current_step)
            if out_criterion["mse_loss_S"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: mse_loss_S'), out_criterion["mse_loss_S"].item(),
                                     current_step)
            if out_criterion["crossview_contrastive_Loss"] is not None:
                tb_logger.add_scalar('{}'.format('[train]: crossview_contrastive_Loss'), out_criterion["crossview_contrastive_Loss"].item(),
                                     current_step)

        if i % 10 == 0:
            # marker = 'ccl' if out_criterion["ms_ssim_loss"] else 'MS-SSIM'
            print(
                f"Train epoch {epoch}: ["
                f"{i * train_dataloader.batch_size:5d}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)] "
                f'Loss: {out_criterion["loss"].item():.3f} | '
                f'mseT loss: {out_criterion["mse_loss_T"].item():.3f} | '
                f'mseS loss: {out_criterion["mse_loss_S"].item():.3f} | '
                f'CCL loss: {out_criterion["crossview_contrastive_Loss"].item():.3f} | '
                f'PSNR: {psrn.mean():.3f} | '
                f'MS-SSIM: {msssim.mean().detach().item():.6f} | '
            )

    logger_train.info(
        f"Train epoch {epoch}: Average losses: "
        f"Loss: {train_loss.avg:.4f} | "
        f"PSNR: {train_psrn.avg:.6f} | "
        f"MS-SSIM: {train_ms_ssim.avg:.6f} |"
        f"CCL: {train_ccl.avg:.6f} |"
        f'mseT loss: {train_mse_T.avg:.6f} | '
        f'mseS loss: {train_mse_S.avg:.6f} | '
    )
    return current_step


def test_epoch(epoch, test_dataloader, model, criterion, save_dir, logger_val, tb_logger):
    model.eval()
    device = next(model.parameters()).device
    keys = ["loss", "mse_loss_S", "mse_loss_T", "crossview_contrastive_Loss", "ms_ssim_loss"]
    meters = {key: AverageMeter() for key in keys}
    psnr, similarity = AverageMeter(), AverageMeter()
    with torch.no_grad():
        for i, (source, target) in enumerate(test_dataloader):
            source, target = source.to(device).float(), target.to(device)
            z_source, z_target, predicted_source, predicted_target = model(source, target)
            out = criterion(z_source, z_target, predicted_source, predicted_target, source, target)
            for key in keys:
                meters[key].update(out[key].item(), source.size(0))
            # Per-image, clipped and uint8-quantized metrics, matching legacy validation.
            for predicted, expected in zip(predicted_target, target):
                p, m = compute_metrics(torch2img(predicted), torch2img(expected))
                psnr.update(p)
                similarity.update(m)
            if (epoch + 1) % 20 == 0:
                from torchvision.utils import save_image
                os.makedirs(save_dir, exist_ok=True)
                save_image(predicted_target, os.path.join(save_dir, f"TargetPre_{i:03d}.png"))
                for name, value in [("SourcePre", predicted_source), ("SourceLat", z_source), ("TargetLat", z_target)]:
                    values = value.reshape(-1, value.size(-1)).detach().cpu().numpy()
                    np.savetxt(os.path.join(save_dir, f"{name}_{i:03d}.txt"), values)
    if meters["loss"].count == 0:
        raise ValueError("Validation loader is empty")
    for key, meter in meters.items():
        tb_logger.add_scalar(f"[val]: {key}", meter.avg, epoch + 1)
    tb_logger.add_scalar("[val]: psnr", psnr.avg, epoch + 1)
    tb_logger.add_scalar("[val]: ms-ssim", similarity.avg, epoch + 1)
    logger_val.info("Validation epoch %d: loss %.6f | tactile MSE %.6f | visual MSE %.6f | CCL %.6f | PSNR %.6f | MS-SSIM %.6f",
                    epoch, meters["loss"].avg, meters["mse_loss_S"].avg, meters["mse_loss_T"].avg,
                    meters["crossview_contrastive_Loss"].avg, psnr.avg, similarity.avg)
    return float(meters["loss"].avg), float(psnr.avg)


def main():
    args = parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    date = str(datetime.datetime.now())
    date = date[:date.rfind(".")].replace("-", "").replace(":", "").replace(" ", "_")
    log_dir = os.path.join('./logs', f"Test_PSNR_{date}")
    os.makedirs(log_dir, exist_ok=True)

    summary_dir = os.path.join(log_dir, "summary")
    os.makedirs(summary_dir, exist_ok=True)
    tb_logger = SummaryWriter(logdir=summary_dir, comment='info')

    setup_logger('base', log_dir, 'global', level=logging.INFO, screen=True, tofile=True)
    logger = logging.getLogger('base')
    logger.info(f'[*] Start Log To {log_dir}')

    # copy code
    dirs_to_make = next(os.walk('./'))[1]
    not_dirs = ['.data', '.checkpoint', 'logs', '.gitignore', '.venv', '__pycache__']
    os.makedirs(os.path.join(log_dir, 'codes'), exist_ok=True)
    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        os.makedirs(os.path.join(log_dir, 'codes', to_make))

    pyfiles = glob("./*.py")
    for py in pyfiles:
        shutil.copyfile(py, os.path.join(log_dir, 'codes') + "/" + py)

    for to_make in dirs_to_make:
        if to_make in not_dirs:
            continue
        tmp_files = glob(os.path.join('./', to_make, "*.py"))
        for py in tmp_files:
            shutil.copyfile(py, os.path.join(log_dir, 'codes', py[2:]))

    with open(os.path.join(log_dir, 'setting.json'), 'w') as f:
        flags_dict = {k: vars(args)[k] for k in vars(args)}
        json.dump(flags_dict, f, indent=4, sort_keys=True, ensure_ascii=False)

    train_dataset = Datasets(args.train_dataset)
    test_dataset = Datasets(args.test_dataset)
    print(args.train_dataset)
    print(args.test_dataset)

    logger.info(f'[*] Train File Account For {len(train_dataset)}, val {len(test_dataset)}')
    # exit()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=True,
    )

    # net = Model()
    net = Model()
    net = net.to(device)
    logger.info(f'[*] Total Parameters = {sum(p.numel() for p in net.parameters() if p.requires_grad)}')

    # if torch.cuda.device_count() > 1:
    #     net = CustomDataParallel(net)
    milestones = [450, 550]
    optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = Loss(metrics=args.metrics)

    if args.checkpoint != '':
        logger.info('Loading %s', args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        if checkpoint.get("revision") != "unet_bidirectional_v2":
            raise ValueError("Legacy checkpoints are not resumable with the new head/Adam objective. Start a new run.")
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        current_step = checkpoint["global_step"]
        best_psnr = checkpoint["best_psnr"]
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0
        best_psnr = -float("inf")
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        current_step = train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            epoch,
            args.clip_max_norm,
            logger,
            tb_logger,
            current_step,
            args
        )

        save_dir = os.path.join(log_dir, 'val_images', '%03d' % (epoch + 1))
        loss, psnr = test_epoch(epoch, test_dataloader, net, criterion, save_dir, logger, tb_logger)
        lr_scheduler.step()

        is_best = best_psnr < psnr
        best_psnr = max(best_psnr, psnr)
        best_loss = min(loss, best_loss)
        state = {
                "revision": "unet_bidirectional_v2",
                "global_step": current_step,
                "best_psnr": best_psnr,
                "epoch": epoch,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
        torch.save(state, os.path.join(log_dir, f"latest.pth"))
        if is_best:
            torch.save(state, os.path.join(log_dir, f"best.pth"))

    tb_logger.close()


if __name__ == "__main__":
    main()

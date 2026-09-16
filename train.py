import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
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
torch.cuda.empty_cache()

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
        msssim = ms_ssim(target, TargetPredict, data_range=1.0)
        # ccl = crossview_contrastive_Loss(SourceLatent, TargetLatent)

        train_loss.update(out_criterion["loss"].item(), target.size(0))
        train_psrn.update(psrn, target.size(0))
        train_ms_ssim.update(msssim.detach().item(), target.size(0))
        train_mse_T.update(out_criterion["mse_loss_T"].detach().item(), target.size(0))
        train_mse_S.update(out_criterion["mse_loss_S"].detach().item(), target.size(0))
        train_ccl.update(out_criterion["crossview_contrastive_Loss"].detach().item(), target.size(0))

        current_step += 1
        if current_step % 100 == 0:
            # tb_logger.add_scalar('{}'.format('[train]: loss'), out_criterion["loss"].item(), current_step)
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
                f"{i * len(d):5d}/{len(train_dataloader.dataset)}"
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

    loss = AverageMeter()
    # mse_loss = AverageMeter()
    ms_ssim_loss = AverageMeter()
    psnr = AverageMeter()
    ms_ssim = AverageMeter()
    CCL = AverageMeter()
    mse_T = AverageMeter()
    mse_S = AverageMeter()

    with torch.no_grad():
        for i, d in enumerate(test_dataloader):
            source = d[0].to(device).float()
            target = d[1].to(device)

            SourceLatent, TargetLatent, SourcePredict, TargetPredict = model(source, target)
            out_criterion = criterion(SourceLatent, TargetLatent, SourcePredict, TargetPredict, source, target)

            loss.update(out_criterion["loss"])
            # if out_criterion["mse_loss"] is not None:
            #     mse_loss.update(out_criterion["mse_loss"])
            if out_criterion["ms_ssim_loss"] is not None:
                ms_ssim_loss.update(out_criterion["ms_ssim_loss"])
            if out_criterion["mse_loss_T"] is not None:
                mse_T.update(out_criterion["mse_loss_T"])
            if out_criterion["mse_loss_S"] is not None:
                mse_S.update(out_criterion["mse_loss_S"])
            if out_criterion["crossview_contrastive_Loss"] is not None:
                CCL.update(out_criterion["crossview_contrastive_Loss"])

            for prediction, ground_truth in zip(TargetPredict, target):
                rec = torch2img(prediction)
                img = torch2img(ground_truth)
                p, m = compute_metrics(rec, img)
                psnr.update(p)
                ms_ssim.update(m)

            if (epoch + 1) % 20 == 0:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if (i + 1) % 10 == 0 or 1:
                    from torchvision.utils import save_image
                    save_image(torch.cat([TargetPredict]), os.path.join(save_dir, 'TargetPre_%03d.png' % i))
                    np.savetxt(os.path.join(save_dir, 'SourcePre%03d.txt' % i), SourcePredict.reshape(-1, SourcePredict.size(-1)).detach().cpu().numpy())
                    np.savetxt(os.path.join(save_dir, 'SourceLat_%03d.txt' % i), SourceLatent.reshape(-1, SourceLatent.size(-1)).detach().cpu().numpy())
                    np.savetxt(os.path.join(save_dir, 'TargetLat_%03d.txt' % i), TargetLatent.reshape(-1, TargetLatent.size(-1)).detach().cpu().numpy())
            # np.savetxt(os.path.join(save_dir, 'Sourceepoch_%03d.txt' % i), SourceLatent.view(-1, SourceLatent.size(-1)).cpu().numpy())
            # np.savetxt(os.path.join(save_dir, 'Targetepoch_%03d.txt' % i), TargetLatent.view(-1, SourceLatent.size(-1)).cpu().numpy())


    tb_logger.add_scalar('{}'.format('[val]: loss'), loss.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: psnr'), psnr.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: ms-ssim'), ms_ssim.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: CCL'), CCL.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: mse_T'), mse_T.avg, epoch + 1)
    tb_logger.add_scalar('{}'.format('[val]: mse_S'), mse_S.avg, epoch + 1)

    logger_val.info(
        f"Test epoch {epoch}: Average losses: "
        f"Loss: {loss.avg:.4f} | "
        f"PSNR: {psnr.avg:.6f} | "
        f"MS-SSIM: {ms_ssim.avg:.6f} |"
        f"CCL: {CCL.avg:.6f} |"
        f'mseT loss: {mse_T.avg:.6f} | '
        f'mseS loss: {mse_S.avg:.6f} | '
    )

    # if out_criterion["mse_loss"] is not None:
    #     tb_logger.add_scalar('{}'.format('[val]: mse_loss'), mse_loss.avg, epoch + 1)
    if out_criterion["ms_ssim_loss"] is not None:
        tb_logger.add_scalar('{}'.format('[val]: ms_ssim_loss'), ms_ssim_loss.avg, epoch + 1)
    if out_criterion["mse_loss_T"] is not None:
        tb_logger.add_scalar('{}'.format('[train]: mse_loss_T'), mse_T.avg, epoch + 1)
    if out_criterion["mse_loss_S"] is not None:
        tb_logger.add_scalar('{}'.format('[train]: mse_loss_S'), mse_S.avg, epoch + 1)
    if out_criterion["crossview_contrastive_Loss"] is not None:
        tb_logger.add_scalar('{}'.format('[train]: crossview_contrastive_Loss'),
                             CCL.avg, epoch + 1)

    return loss.avg, psnr.avg


def main():
    device = "cuda"
    args = parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)

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
        logger.info("Loading %s", args.checkpoint)
        checkpoint = torch.load(args.checkpoint, map_location=device)
        net.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        current_step = start_epoch * math.ceil(len(train_dataloader.dataset) / args.batch_size)
    else:
        start_epoch = 0
        best_loss = 1e10
        current_step = 0

    best_psnr = checkpoint.get("best_psnr", -float("inf")) if args.checkpoint else -float("inf")
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
                "epoch": epoch,
                "best_psnr": best_psnr,
                "state_dict": net.state_dict(),
                "loss": loss,
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
            }
        torch.save(state, os.path.join(log_dir, f"latest.pth"))
        if is_best:
            torch.save(state, os.path.join(log_dir, f"best.pth"))


if __name__ == "__main__":
    main()

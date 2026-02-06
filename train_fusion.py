import os
import kornia
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import datetime
import utils
from data_RGB import get_training_data
from FusionNet import FusionNet
from warmup_scheduler import GradualWarmupScheduler
from tqdm import tqdm
import warnings
from torch.utils.tensorboard import SummaryWriter
from ssim1 import *
import logging
import argparse
import clip
from utils.computer_metric1 import metric
# Argument parser for configuration
def parse_args():
    parser = argparse.ArgumentParser(description='Training configuration')

    # Dataset and training paths
    parser.add_argument('--train_dir', type=str, default='./dataset/train/whu', help='Path to training data')
    parser.add_argument('--val_dir', type=str, default='./dataset/test/whu', help='Path to validation data')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/whu', help='Path to save models and images')
    parser.add_argument('--save_val_dir', type=str, default='./results_val/whu', help='Path to save validation and images')

    # Optimization arguments
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--lr_initial', type=float, default=2e-5, help='Initial learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-6, help='Minimum learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Beta1 for Adam optimizer')

    # Validation and resuming
    parser.add_argument('--val_after_every', type=int, default=5, help='Validate after every n epochs')
    parser.add_argument('--resume', type=bool, default=False, help='Resume training from checkpoint')

    # GPU settings
    parser.add_argument('--gpu', type=str, default='0', help='Comma separated list of GPU ids to use')

    return parser.parse_args()

# Convert RGB to YCbCr
def rgb_to_ycbcr(img):
    return kornia.color.rgb_to_ycbcr(img)

# Main training function
def main():
    # Parse arguments
    args = parse_args()
    utils.mkdir(args.save_dir)
    utils.mkdir(args.save_val_dir)

    # Setup GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    torch.backends.cudnn.benchmark = True

    # Create necessary directories
    log_file_path = './logs/train/training_fmb.txt'
    log_folder = os.path.dirname(log_file_path)
    os.makedirs(log_folder, exist_ok=True)
    logging.basicConfig(filename=log_file_path, level=logging.INFO)
    logger = logging.getLogger(__name__)
    file_handler = logging.FileHandler(log_file_path)
    logger.addHandler(file_handler)

    # Set up TensorBoard logging
    current_datetime = datetime.datetime.now()
    date_time_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join("logs", "training_fmb", date_time_str)
    writer = SummaryWriter(log_dir)

    ######### Set Seeds ###########
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Model setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ######### Model ###########
    model_clip, _ = clip.load("RN50", device=device)
    model = FusionNet(model_clip).cuda()
    if torch.cuda.device_count() > 1:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    # Optimizer setup
    optimizer = optim.Adam(model.parameters(), lr=args.lr_initial, betas=(args.beta1, 0.999), eps=1e-8)

    # Scheduler setup
    warmup_epochs = 3
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.num_epochs - warmup_epochs, eta_min=args.lr_min)
    scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    scheduler.step()

    # Resume setup
    start_epoch = 0
    if args.resume:
        path_chk_rest = utils.get_last_path(args.save_dir, 'model_latest.pth')
        utils.load_checkpoint(model, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest)
        utils.load_optim(optimizer, path_chk_rest)
        for i in range(1, start_epoch):
            scheduler.step()
        print('Resuming Training with learning rate:', scheduler.get_lr()[0])

    ######### Loss ###########
    fus_criterion = Fusionloss()

    ######### DataLoaders ###########
    train_dataset = get_training_data(args.train_dir)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)
    val_dataset = get_training_data(args.val_dir)
    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, drop_last=True, pin_memory=True)
    # Logging
    fusion_loss_log = 0
    train_step = 0
    for epoch in range(start_epoch, args.num_epochs + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        model.train()

        for i, data in enumerate(tqdm(train_loader), 0):
            for param in model.parameters():
                param.grad = None

            input_ir = data[0].cuda()
            input_rgb = data[1].cuda()
            input_ycbcr = rgb_to_ycbcr(input_rgb)
            fusion = model(input_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])
            # fusion_ycbcr = torch.cat((torch.clamp(fusion, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]), dim=1)
            # fusion_res = kornia.color.ycbcr_to_rgb(fusion_ycbcr)

            loss_fusion = fus_criterion(input_ycbcr[:, :1, :, :], input_ir[:, :1, :, :], fusion)

            loss = loss_fusion
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            fusion_loss_log += loss.item()

        lenth = len(train_loader)
        avg_fusion_loss = fusion_loss_log / lenth
        writer.add_scalar('fusion_loss_log/train', avg_fusion_loss, train_step)
        train_step += 1
        fusion_loss_log = 0

        #### Evaluation ####
        if epoch % args.val_after_every == 0:
            model.eval()
            metric_result = np.zeros((8))
            for ii, data_val in enumerate((val_loader), 0):
                input_ir = data_val[0].cuda()
                input_rgb = data_val[1].cuda()
                filenames = data_val[2]
                input_ycbcr = rgb_to_ycbcr(input_rgb)
                with torch.no_grad():
                    fusion = model(input_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])
                fusion = torch.clamp(fusion, 0, 1)
                res_ycbcr = torch.cat((torch.clamp(fusion, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
                                      dim=1)
                res_rgb = kornia.color.ycbcr_to_rgb(res_ycbcr)

                ones = torch.ones_like(res_rgb)
                zeros = torch.zeros_like(res_rgb)

                res_rgb = torch.where(res_rgb > ones, ones, res_rgb)
                res_rgb = torch.where(res_rgb < zeros, zeros, res_rgb)
                res_rgb = res_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()
                res_rgb = np.uint8(255.0 * res_rgb)
                # -----------------------------------------------------#
                # -----save ir, rgb------#
                for batch in range(len(fusion)):
                    rgb_img = res_rgb[batch]
                    utils.save_img((os.path.join(args.save_val_dir, filenames[batch] + '.png')), rgb_img)

            ir_dir = os.path.join(args.val_dir, 'ir')
            vi_dir = os.path.join(args.val_dir, 'rgb')
            fus_dir = args.save_val_dir

            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       os.path.join(args.save_dir, f"model_epoch_{epoch}.pth"))
        scheduler.step()

        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.8f}".format(epoch, time.time() - epoch_start_time,
                                                                                  epoch_loss, scheduler.get_lr()[0]))
        print("fusion_loss: {:.4f}".format(avg_fusion_loss))
        print("------------------------------------------------------------------")

        logging.info(f'Epoch: {epoch}\tTime: {time.time() - epoch_start_time:.4f}\tLoss: {epoch_loss:.4f}\tLearningRate {scheduler.get_lr()[0]:.8f}\n')
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   os.path.join(args.save_dir, "model_latest.pth"))

    writer.close()
    file_handler.close()

if __name__ == '__main__':
    main()

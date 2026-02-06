import kornia
import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import utils

from data_RGB import get_test_data
from FusionNet_multiprompt_3 import FusionNet
import clip
import time
from utils.seg_util import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def rgb_to_ycbcr(img):
    ycbcr = kornia.color.rgb_to_ycbcr(img)
    return ycbcr

parser = argparse.ArgumentParser(description='Image Fusion using FusionNet')
#../ReconstructNet/dataset/train/mfnet
parser.add_argument('--input_dir', default='./dataset/test/potsdam', type=str, help='Directory for results')
parser.add_argument('--result_dir', default='./results/potsdam_multiprompt3_div10', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/potsdam_multiprompt3_div10/model_epoch_20.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

######### Model ###########
device = "cuda" if torch.cuda.is_available() else "cpu"
######### Model ###########
model_clip, _ = clip.load("RN50", device=device)
model = FusionNet(model_clip).cuda()
# fusion_model = FusionNet()
model.cuda()

utils.load_checkpoint(model, args.weights)
print("===>Testing using weights: ",args.weights)
model.cuda()
model = nn.DataParallel(model)
model.eval()

test_dataset = get_test_data(args.input_dir)
test_loader = DataLoader(dataset=test_dataset, batch_size=2, shuffle=False, num_workers=4, drop_last=False,
                         pin_memory=True)

utils.mkdir(args.result_dir)
with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        torch.cuda.ipc_collect()
        input_ir = data_test[0].cuda()
        input_rgb = data_test[1].cuda()
        filenames = data_test[2]

        input_hsv = kornia.color.rgb_to_hsv(input_rgb)
        res,_ = model(input_ir[:, :1, :, :], input_hsv[:, 2:, :, :])
        # ------图像重建------#
        res = torch.clamp(res, 0, 1)
        res_hsv = torch.cat((input_hsv[:, 0:1, :, :], input_hsv[:, 1:2, :, :], res), dim=1)
        res_rgb = kornia.color.hsv_to_rgb(res_hsv)

        ones = torch.ones_like(res_rgb)
        zeros = torch.zeros_like(res_rgb)
        # -----generation ir and rgb from [0,1] to [0,255]-----#
        res_rgb = torch.where(res_rgb > ones, ones, res_rgb)
        res_rgb = torch.where(res_rgb < zeros, zeros, res_rgb)
        res_rgb = res_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()
        res_rgb = np.uint8(255.0 * res_rgb)

        # input_ycbcr = rgb_to_ycbcr(input_rgb)
        # fusion = model(input_ir[:, :1, :, :], input_ycbcr[:, :1, :, :])
        #
        # fus_ycbcr = torch.cat((torch.clamp(fusion, 0, 1), input_ycbcr[:, 1:2, :, :], input_ycbcr[:, 2:, :, :]),
        #                       dim=1)
        # fus_rgb = kornia.color.ycbcr_to_rgb(fus_ycbcr)
        # ones = torch.ones_like(fusion)
        # zeros = torch.zeros_like(fusion)
        # fus_rgb = torch.where(fus_rgb > ones, ones, fus_rgb)
        # fus_rgb = torch.where(fus_rgb < zeros, zeros, fus_rgb)
        # fus_rgb = fus_rgb.permute(0, 2, 3, 1).cpu().detach().numpy()  # B C H W->B H W C
        # fus_rgb = (fus_rgb - np.min(fus_rgb)) / (
        #         np.max(fus_rgb) - np.min(fus_rgb)
        # )
        # fus_rgb = np.uint8(255.0 * fus_rgb)

        # -----------------------------------------------------#
        for batch in range(len(res_rgb)):
            rgb_img = res_rgb[batch]
            utils.save_img((os.path.join(args.result_dir, filenames[batch] + '.png')), rgb_img)





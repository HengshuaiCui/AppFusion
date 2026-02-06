import torch
import torch.nn as nn
import kornia.losses as losses
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from loss_ssim import SSIMLoss
import kornia.losses as losses
# from SSIM import *
# from MEF_SSIM_loss import th_SSIM_LOSS
ssim_loss = SSIMLoss(window_size=7)
SSIMLOSS = losses.SSIMLoss(window_size=11, reduction='mean')
# SSIM = SSIM()

import torch
import torch.nn as nn

class Fusionloss(nn.Module):
    def __init__(self):
        super(Fusionloss, self).__init__()
        # self.sobelconv = Sobelxy_cuda0()

    def forward(self, image_vis, image_ir, generate_img):

        # 计算tensor的均值
        s_loss_o = 1-ssim_loss(image_vis,generate_img)
        s_loss_i = 1-ssim_loss(image_ir,generate_img)
        # s_loss_o = SSIMLOSS(generate_img,image_vis)
        # s_loss_i = SSIMLOSS(generate_img,image_ir)
        s_loss = s_loss_o + s_loss_i

        loss_total = s_loss
        return loss_total

class Fusionloss_mfnet(nn.Module):
    def __init__(self):
        super(Fusionloss_mfnet, self).__init__()
        self.sobelconv = Sobelxy_cuda0()

    def forward(self, image_vis, image_ir, generate_img):
        C = 0.0001
        hyp_prm = 1.7
        # μ_yk
        y1_mean = torch.mean(image_vis)
        y2_mean = torch.mean(image_ir)
        # y_k upperwave
        y1_mean_sub = image_vis - y1_mean
        y2_mean_sub = image_ir - y2_mean
        # c_k and c^(c_upperArrow)
        c1 = torch.norm(y1_mean_sub)
        c2 = torch.norm(y2_mean_sub)
        c_upperArrow = torch.maximum(c1, c2)
        # wmygfh
        c_upperArrow *= hyp_prm

        # s_k
        s1 = y1_mean_sub / (c1 + C)
        s2 = y2_mean_sub / (c2 + C)

        # s upper dash
        s_upperDash = s1 + s2
        # s_upperDash = tf.maximum(s1, s2)
        # s^
        s_upperArrow = s_upperDash / (torch.norm(s_upperDash) + C)

        # y^
        y_upperArrow = c_upperArrow * s_upperArrow

        loss_total = 1 - ssim_loss(y_upperArrow, generate_img)
        # loss_total = 1-ssim_loss(y_upperArrow,generate_img)
        # loss_total = SSIM(y_upperArrow, generate_img) gray1
        return loss_total


class Fusionloss_fmb_1(nn.Module):
    def __init__(self):
        super(Fusionloss_fmb_1, self).__init__()
        self.sobelconv = Sobelxy_cuda0()

    def forward(self, image_vis, image_ir, generate_img):


        target=0.5*image_ir +0.5*image_vis

        loss_total = 1-ssim_loss(target,generate_img)

        return loss_total

class Fusionloss_meta(nn.Module):
    def __init__(self):
        super(Fusionloss_meta, self).__init__()
        self.sobelconv = Sobelxy_cuda0()

    def forward(self, image_vis, image_ir, generate_img):

        s_loss_o = 1-ssim_loss(image_vis,generate_img)
        s_loss_i = 1-ssim_loss(image_ir,generate_img)
        s_loss = s_loss_o + 0.1 * s_loss_i

        loss_total = s_loss
        return loss_total

class Fusionloss_sar(nn.Module):
    def __init__(self):
        super(Fusionloss_sar, self).__init__()
        self.sobelconv = Sobelxy_cuda0()

    def forward(self, image_vis, image_ir, generate_img):

        # 计算tensor的均值
        s_loss_o = 1-ssim_loss(image_vis,generate_img)
        s_loss_i = 1-ssim_loss(image_ir,generate_img)

        s_loss = s_loss_o + s_loss_i

        loss_total = s_loss
        return loss_total

class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda(1)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda(1)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

class Sobelxy_cuda0(nn.Module):
    def __init__(self):
        super(Sobelxy_cuda0, self).__init__()
        kernelx = [[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]]
        kernely = [[1, 2, 1],
                   [0, 0, 0],
                   [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda(0)
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda(0)

    def forward(self, x):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx) + torch.abs(sobely)

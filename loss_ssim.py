import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp
from PIL import Image
import matplotlib.pyplot as plt
import math
import cv2
import numpy as np


"""
# ============================================
# SSIM loss
# https://github.com/Po-Hsun-Su/pytorch-ssim
# ============================================
"""


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2
# 4 8 2 4
    sigma1_sq = 8*(F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq)##20.25 2 8
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = 3*(F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2)#4.5  2  4

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def Contrast(img1, img2, window_size=7, channel=1):
    window = create_window(window_size, channel)    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq

    return sigma1_sq, sigma2_sq

    
class SSIMLoss(torch.nn.Module):
    def __init__(self, window_size=7, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)


def ssim(img1, img2, window_size=7, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


if __name__ == '__main__':
    import cv2
    from torch import optim
    from skimage import io
    npImg1 = cv2.imread("/media/yt/DIsk2/wwb/multi_task_fusion/data_sar_cut/ir_test/114.png")
    npImg2 = cv2.imread("/media/yt/DIsk2/wwb/multi_task_fusion/data_sar_cut/rgb_test/114.png")


    img1 = torch.from_numpy(np.rollaxis(npImg1, 2)).float().unsqueeze(0)/255.0
    img2 = torch.from_numpy(np.rollaxis(npImg2, 2)).float().unsqueeze(0)/255.0
    img3 = torch.rand(img1.size())


    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()
        img3 = img3.cuda()

    img1 = Variable(img1, requires_grad=False)
    img2 = Variable(img2, requires_grad=False)
    img3 = Variable(img3, requires_grad=True)

    ssim_value_1 = ssim(img1, img3).item()
    ssim_value_2 = ssim(img2, img3).item()
    ssim_value = 2-ssim_value_1 - ssim_value_2
    print("Initial ssim:", ssim_value)

    ssim_loss = SSIMLoss()
    optimizer = optim.Adam([img3], lr=0.01)

    while ssim_value > 0.2195:
        optimizer.zero_grad()
        s_loss = (1-ssim_loss(img1, img3)) + (1-ssim_loss(img2, img3))
        # s_loss = 1-ssim_loss(img2, img3)
        ssim_value = s_loss.item()
        print('{:<4.4f}'.format(ssim_value))
        s_loss.backward()
        optimizer.step()

    img_1 = np.transpose(img1.detach().cpu().squeeze().float().numpy(), (1,2,0))
    img_1 = np.uint8(np.clip(img_1*255, 0, 255))

    img_2 = np.transpose(img2.detach().cpu().squeeze().float().numpy(), (0,1,2))
    img_2 = np.uint8(np.clip(img_2*255, 0, 255))

    img_3 = np.transpose(img3.detach().cpu().squeeze().float().numpy(), (1,2,0))
    img_3 = np.uint8(np.clip(img_3*255, 0, 255))


    def calculate_entropy(img):
        # 将图像数组转换为整数类型
        img = img.astype(np.uint8)
        # 计算直方图
        hist, _ = np.histogram(img, bins=256, range=(0, 256))
        # 计算概率分布
        prob = hist / float(hist.sum())
        # 计算熵
        entropy = -np.sum(prob * np.log2(prob + np.finfo(float).eps))
        return entropy



    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_1)
    axes[0].set_title("Input RGB")
    axes[0].axis("off")
    axes[1].imshow(npImg2)
    axes[1].set_title("Output RGB")
    axes[1].axis("off")
    axes[2].imshow(img_3)
    axes[2].set_title("Output RGB")
    axes[2].axis("off")

    plt.show()

    print("en:", calculate_entropy(img_1))
    print("en:", calculate_entropy(img_2))
    print("en:", calculate_entropy(img_3))

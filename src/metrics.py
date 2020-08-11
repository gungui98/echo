import torch
import math


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * math.log10(pixel_max / math.sqrt(mse))

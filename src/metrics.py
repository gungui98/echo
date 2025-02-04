import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp, log10, sqrt


def psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    pixel_max = 1.0
    return 20 * log10(pixel_max / sqrt(mse))


# https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
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


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def iou_metric_segmentation(pred, gt):
    """
    IoU metric for segmentation task with num_classes
    :arg:
        pred: torch.Tensor, [N, C, H, W]
        gt: torch.Tensor, [N, C, H, W]
    """
    assert pred.size() == gt.size(), "pred and gt should have same size"
    assert pred.dim() == 4, "pred should be 4D, got {}".format(pred.dim())
    assert gt.dim() == 4, "gt should be 4D, got {}".format(gt.dim())
    assert pred.size(1) == gt.size(1), "pred and gt should have same channel"

    pred = pred.reshape(pred.size(0), -1)
    gt = gt.reshape(pred.size(0), -1)
    intersection = (pred * gt).sum(dim=1)
    union = pred.sum(dim=1) + gt.sum(dim=1) - intersection
    iou = (intersection + 1e-7) / (union + 1e-7)
    return iou.mean()


if __name__ == '__main__':
    labels = torch.zeros(16, 1, 256, 256)
    labels[:, :, 0:128, 0:128] = 1
    preds = torch.zeros(16, 1, 256, 256)
    preds[:, :, 64:192, 64:192] = 1
    # show image and label
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(labels[0, 0, :, :])
    plt.subplot(122)
    plt.imshow(preds[0, 0, :, :])
    plt.show()
    print(iou_metric_segmentation(preds, labels))

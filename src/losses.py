import torch
from torch import nn
from transformers import BertModel


class MSELossWithMask(nn.Module):
    def __init__(self, mask_value=0):
        super(MSELossWithMask, self).__init__()

    def forward(self, input, target, mask):
        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        input = input * mask
        target = target * mask
        return torch.mean(torch.pow(input - target, 2))


class L1LossWithMask(nn.Module):
    def __init__(self, mask_value=0):
        super(L1LossWithMask, self).__init__()

    def forward(self, input, target, mask):
        """

        :param input: BxCxHxW
        :param target: BxCxHxW
        :param mask: Bx1
        :return:
        """
        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        input = input * mask
        target = target * mask
        return torch.mean(torch.abs(input - target))

class BCELossWithMask(nn.Module):
    def __init__(self, mask_value=0):
        super(BCELossWithMask, self).__init__()

    def forward(self, input, target, mask):
        batch_size = input.shape[0]
        input = input.view(batch_size, -1)
        target = target.view(batch_size, -1)
        mask = mask.view(batch_size, -1)
        input = input * mask
        target = target * mask
        return torch.mean(torch.nn.functional.binary_cross_entropy_with_logits(input, target))

if __name__ == '__main__':
    pred = torch.randn(16, 1, 128, 128)
    target = torch.randn(16, 1, 128, 128)
    loss_function = L1LossWithMask()
    loss = loss_function(pred, target, torch.ones(16, 1))
    print(loss)
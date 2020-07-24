from torchvision import transforms
import skimage.io as io
from PIL import Image
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from glob import glob

SEED = 42  # fix it


class DatasetCAMUS(Dataset):
    def __init__(self, dataset_path, input_name, target_name, condition_name,
                 img_res, target_rescale, input_rescale, condition_rescale,
                 train_ratio, valid_ratio, labels, augment, subset='train'):

        self.dataset_path = dataset_path
        self.img_res = tuple(img_res)
        self.target_rescale = target_rescale
        self.input_rescale = input_rescale
        self.condition_rescale = condition_rescale
        self.input_name = input_name
        self.target_name = target_name
        self.condition_name = condition_name
        self.augment = augment
        self.items = []

        patients = sorted(glob(os.path.join(self.dataset_path, 'training', '*')))

        for patient in patients:
            path = patient
            head, patient_id = os.path.split(path)
            target_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.target_name))
            condition_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.condition_name))
            input_path = os.path.join(path, '{}_{}.mhd'.format(patient_id, self.input_name))
            self.items.append((target_path, condition_path, input_path))

        random.Random().shuffle(self.items)  # one seed for all files?
        num = len(self.items)

        all_labels = {0, 1, 2, 3}
        self.not_labels = all_labels - set(labels)

        if subset == 'train':
            self.items = self.items[: int(train_ratio * num)]
        elif subset == 'valid':
            self.items = self.items[int(train_ratio * num): int(train_ratio * num) + int(valid_ratio * num)]
        elif subset == 'test':
            self.items = self.items[int(train_ratio * num) + int(valid_ratio * num):]

    def read_mhd(self, img_path, is_gt):

        # if not os.path.exists(img_path): # empty folders
        #    print(img_path)
        #    #return np.zeros(self.img_res + (1,))
        #    return np.zeros(self.img_res)
        img = io.imread(img_path, plugin='simpleitk').squeeze()

        img = np.array(Image.fromarray(img).resize(self.img_res))

        if is_gt:
            for not_l in self.not_labels:
                img[img == not_l] = 0
        return img

    def get_weight_map(self, mask):
        # let the y axis have higher variance
        gauss_var = [[self.img_res[0] * 60, 0], [0, self.img_res[1] * 30]]
        # print(mask.shape) # (256, 256) or (256, 256, 1) for empty
        # if len(mask.shape) > 2:
        #    mask = np.squeeze(mask, axis=2)
        #
        x, y = mask.nonzero()

        center = [x.mean(), y.mean()]

        from scipy.stats import multivariate_normal
        gauss = multivariate_normal.pdf(np.mgrid[
                                        0:self.img_res[1],
                                        0:self.img_res[0]].reshape(2, -1).transpose(),
                                        mean=center,
                                        cov=gauss_var)

        gauss /= gauss.max()
        gauss = gauss.reshape((self.img_res[1], self.img_res[0]))

        # set the gauss value of the main target part to 1
        gauss[mask > 0] = 1

        return gauss

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        target_path, condition_path, input_path = self.items[index]

        target = self.read_mhd(img_path=target_path, is_gt=True)
        condition = self.read_mhd(img_path=condition_path, is_gt=True)
        input_ = self.read_mhd(img_path=input_path, is_gt=False)

        weight_map_condition = self.get_weight_map(condition)

        # to tensor without normalization
        weight_map_condition = torch.tensor(np.asarray(weight_map_condition)).float().unsqueeze(dim=0)
        input_ = torch.tensor(np.asarray(input_)).float().unsqueeze(dim=0)
        condition = torch.tensor(np.asarray(condition)).float().unsqueeze(dim=0)

        target = transforms.ToTensor()(target)

        return target, condition, input_, weight_map_condition

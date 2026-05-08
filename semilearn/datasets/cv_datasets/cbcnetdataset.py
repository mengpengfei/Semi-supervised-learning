import os

import math
import torchvision.io
from PIL.Image import Image
from torch.utils.data import Dataset

from semilearn.datasets.augmentation import RandomResizedCropAndInterpolation, RandAugment
from torchvision.transforms import transforms
import numpy as np


mean, std = {}, {}
mean['imagenet'] = [0.485, 0.456, 0.406]
std['imagenet'] = [0.229, 0.224, 0.225]

def get_cbcnet(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):

    img_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_medium = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(1, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize((int(math.floor(img_size / crop_ratio)), int(math.floor(img_size / crop_ratio)))),
        RandomResizedCropAndInterpolation((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 10),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(math.floor(int(img_size / crop_ratio))),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean['imagenet'], std['imagenet'])
    ])
    lb_dset = CBCNetDataset(txt=os.path.join(data_dir, "train.txt"), transform=transform_weak, is_ulb=False, alg=alg, num_classes=num_classes)
    # lb_dset = CBCNetDataset(root=os.path.join(data_dir, "train"), transform=transform_weak, ulb=False, alg=alg, percentage=percentage)
    ulb_dset = CBCNetDataset(txt=os.path.join(data_dir, "train-ulb.txt"), transform=transform_weak, is_ulb=True, alg=alg, num_classes=num_classes, medium_transform=transform_medium, strong_transform=transform_strong)

    eval_dset = CBCNetDataset(txt=os.path.join(data_dir, "val.txt"), transform=transform_val, is_ulb=False, alg=alg, num_classes=num_classes)

    return lb_dset, ulb_dset, eval_dset




class CBCNetDataset(Dataset):
    """
    CBCNetDataset
    """
    def __init__(self, alg, txt, num_classes, transform, is_ulb, strong_transform=None,medium_transform=None
                 ):
        super(CBCNetDataset, self).__init__()
        # self.is_random_enhance = is_random_enhance
        self.medium_transform=medium_transform
        # if std is None:
        #     std = [0.229, 0.224, 0.225]
        # if mean is None:
        #     mean = [0.485, 0.456, 0.406]
        self.alg = alg
        self.txt = txt
        self.num_classes = num_classes
        self.transform = transform
        self.is_ulb = is_ulb
        self.strong_transform = strong_transform
        self.MEAN = mean
        self.STD = std
        # transform_f=[]
        # if self.is_random_enhance:
        #     transform_f.append(transforms.RandomApply(
        #         [
        #             transforms.ColorJitter(brightness=[0.5,1.2]),
        #             transforms.ColorJitter(contrast=[0.7, 1.3]),
        #             transforms.ColorJitter(saturation=[0.8, 1.2]),
        #             transforms.RandomAffine(degrees=2, translate=(0, 0.2), scale=(0.9, 1), shear=(6, 9), fill=(245,245,244)),
        #             # transforms.RandomPerspective(distortion_scale=1, p=1, interpolation=3),
        #             # transforms.RandomVerticalFlip(),
        #             # transforms.RandomHorizontalFlip(),
        #             # transforms.RandomRotation(10, interpolation=InterpolationMode.BILINEAR,expand=False,center=None),
        #             transforms.RandomAutocontrast(p=0.5),
        #             transforms.RandomGrayscale(p=0.7)
        #         ], p=0.5))
        self.lines=[]
        with open(self.txt, 'r') as f:
            self.lines = f.readlines()

        if self.strong_transform is None:
            if self.is_ulb:
                assert self.alg not in ['fullysupervised', 'supervised', 'pseudolabel', 'vat', 'pimodel', 'meanteacher', 'mixmatch', 'refixmatch'], f"alg {self.alg} requires strong augmentation"

        if self.medium_transform is None:
            if self.is_ulb:
                assert self.alg not in ['sequencematch'], f"alg {self.alg} requires medium augmentation"

    def __sample__(self, idx):
        """ dataset specific sample function """
        # set idx-th target
        line=self.lines[idx]
        line_arr=line.strip().split(' ')
        if len(line_arr)<=1:
            target = None
        else:
            # target = int(line_arr[-1])
            target = np.int64(line_arr[-1])
            # target = get_onehot(self.num_classes, target_)

        img_path = str(line[0])
        img = Image.open(img_path).convert('RGB')
        # img_array = np.array(img)
        # set augmented images
        return img, target

    def __getitem__(self, idx):
        img, target = self.__sample__(idx)
        if self.transform is None:
            return  {'x_lb':  transforms.ToTensor()(img), 'y_lb': target}
        else:
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            img_w = self.transform(img)
            if not self.is_ulb:
                return {'idx_lb': idx, 'x_lb': img_w, 'y_lb': target}
            else:
                if self.alg == 'fullysupervised' or self.alg == 'supervised':
                    return {'idx_ulb': idx}
                elif self.alg == 'pseudolabel' or self.alg == 'vat':
                    return {'idx_ulb': idx, 'x_ulb_w':img_w}
                elif self.alg == 'pimodel' or self.alg == 'meanteacher' or self.alg == 'mixmatch':
                    # NOTE x_ulb_s here is weak augmentation
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.transform(img)}
                # elif self.alg == 'sequencematch' or self.alg == 'somematch':
                elif self.alg == 'sequencematch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_m': self.medium_transform(img), 'x_ulb_s': self.strong_transform(img)}
                elif self.alg == 'remixmatch':
                    rotate_v_list = [0, 90, 180, 270]
                    rotate_v1 = np.random.choice(rotate_v_list, 1).item()
                    img_s1 = self.strong_transform(img)
                    img_s1_rot = torchvision.transforms.functional.rotate(img_s1, rotate_v1)
                    img_s2 = self.strong_transform(img)
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': img_s1, 'x_ulb_s_1':img_s2, 'x_ulb_s_0_rot':img_s1_rot, 'rot_v':rotate_v_list.index(rotate_v1)}
                elif self.alg == 'comatch':
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s_0': self.strong_transform(img), 'x_ulb_s_1':self.strong_transform(img)}
                else:
                    return {'idx_ulb': idx, 'x_ulb_w': img_w, 'x_ulb_s': self.strong_transform(img)}

    def __len__(self):
        return len(self.lines)
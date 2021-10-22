import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor
from PIL import Image
from scipy.signal import convolve2d
import numpy as np
import h5py
import math
from pylab import * 


def default_loader(path):
    return Image.open(path).convert('RGB')


def gray_loader(path):
    return Image.open(path).convert('L')


def LocalNormalization(patch, P=3, Q=3, C=1):
    kernel = np.ones((P, Q)) / (P * Q)
    patch_mean = convolve2d(patch, kernel, boundary='symm', mode='same')
    patch_sm = convolve2d(np.square(patch), kernel, boundary='symm', mode='same')
    patch_std = np.sqrt(np.maximum(patch_sm - np.square(patch_mean), 0)) + C
    patch_ln = torch.from_numpy((patch - patch_mean) / patch_std).float().unsqueeze(0)
    return patch_ln


def OverlappingCropPatches(im, patch_size=64, stride=64):
    w, h = im.size
    patches1 = ()

    for i in range(0, h - stride, stride):
        for j in range(0, w - stride, stride):
            patch = to_tensor(im.crop((j, i, j + patch_size, i + patch_size)))

            # patch = LocalNormalization(patch[0].numpy())  # 归一化
            patches1 = patches1 + (patch,)

    # print('num patches', len(patches1))#num patches 285
    return patches1


class IQADataset(Dataset):
    def __init__(self, conf, index, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['TID2013']['im_dir']
        im_ref = conf['TID2013']['im_ref']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['TID2013']['datainfo']

        Info = h5py.File(datainfo)
        index = Info['index'][:, 0]
        # index = [7, 2, 1, 22, 28, 21, 13, 23, 17, 25, 20, 16, 6, 0, 24, 18, 14, 26, 8, 3, 15, 4, 9, 5, 12, 19, 10, 11, 27]
        # index = [25,19,7,18,2,6,23,26,21,15,13,9,10,24,22,3,8,12,27,14,16,20,0,4,11,28,5,1]
        # index = [17, 8, 16, 13, 5, 4, 2, 27, 6, 23, 14, 25, 7, 0, 21, 10, 12, 28, 18, 3, 1, 15, 19, 26, 24, 20, 9, 11, 22]
        # index = [13, 11, 10, 2, 9, 18, 4, 0, 24, 12, 25, 3, 1, 7, 14, 26, 16, 19, 8, 21, 23, 20, 6]
        ref_ids = Info['ref_ids'][0, :]
        # ref_ids = ref_ids[0:30]

        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = Info['subjective_scores'][0, self.index]
        self.mos_std = Info['subjective_scoresSTD'][0, self.index]

        im_names = [Info[Info['im_names'][0, :][i]].value.tobytes() \
                        [::2].decode() for i in self.index]
        # im_refnames = [Info[Info['ref_names'][0, :][i]].value.tobytes()[::2].decode()
        #              for i in (ref_ids[self.index] - 1).astype(int)]
        ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
                     for i in (ref_ids[self.index] - 1).astype(int)]

        self.patches1 = ()

        self.label = []
        self.refpatches = ()

        for idx in range(len(self.index)):
            # print("Preprocessing Image: {} {} {}".format(im_names[idx], self.mos[idx], ref_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            refimg = self.loader(os.path.join(im_ref, ref_names[idx]))

            # patches1 = OverlappingCropPatches(im, self.patch_size,self.stride)  ################
            # refpatches = OverlappingCropPatches(refimg, self.patch_size, self.stride)
            if status == 'train':
                patches1 = OverlappingCropPatches(im, self.patch_size, self.stride)  ################
                refpatches = OverlappingCropPatches(refimg, self.patch_size, self.stride)

                self.patches1 = self.patches1 + patches1
                self.refpatches = self.refpatches + refpatches

                for i in range(len(patches1)):
                    self.label.append(self.mos[idx])

            else:
                patches1 = OverlappingCropPatches(im, 64, 64)
                refpatches = OverlappingCropPatches(refimg, 64, 64)

                self.patches1 = self.patches1 + (torch.stack(patches1),)
                self.refpatches = self.refpatches + (torch.stack(refpatches),)
                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches1)

    def __getitem__(self, idx):

        return (self.patches1[idx],
                (torch.Tensor([self.label[idx], ]),
                 self.refpatches[idx]))


class IQADataset_all(Dataset):
    def __init__(self, conf, index, exp_id=0, status='train', loader=default_loader):
        self.loader = loader
        im_dir = conf['LIVE']['im_dir']
        im_ref = conf['LIVE']['im_ref']
        self.patch_size = conf['patch_size']
        self.stride = conf['stride']
        datainfo = conf['LIVE']['datainfo']

        Info = h5py.File(datainfo)
        index = Info['index'][:, int(exp_id) % 1000]

        ref_ids = []
        for line0 in open("./data/ref_ids.txt", "r"):
            line0 = float(line0[:-1])
            ref_ids.append(line0)
        ref_ids = np.array(ref_ids)
        # ref_ids = ref_ids[0:10]
        # print(ref_ids)

        test_ratio = conf['test_ratio']
        train_ratio = conf['train_ratio']
        trainindex = index[:int(train_ratio * len(index))]
        testindex = index[int((1 - test_ratio) * len(index)):]
        train_index, val_index, test_index = [], [], []

        # print('trainindex:', trainindex)
        # print('testindex:', testindex)
        # print('testindex:', testindex)
        # print('len(ref_ids)',len(ref_ids))
        for i in range(len(ref_ids)):
            train_index.append(i) if (ref_ids[i] in trainindex) else \
                test_index.append(i) if (ref_ids[i] in testindex) else \
                    val_index.append(i)
        if status == 'train':
            self.index = train_index
            print("# Train Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(trainindex)
        if status == 'test':
            self.index = test_index
            print("# Test Images: {}".format(len(self.index)))
            print('Ref Index:')
            print(testindex)
        if status == 'val':
            self.index = val_index
            print("# Val Images: {}".format(len(self.index)))

        self.mos = []
        for line5 in open("./data/mos.txt", "r"):
            line5 = float(line5.strip())
            self.mos.append(line5)
        self.mos = np.array(self.mos)

        # self.scale = self.mos.max()
        # print('self.scale', self.scale)
        # self.mos = self.mos/self.scale

        im_names = []
        ref_names = []

        for line1 in open("./data/im_names.txt", "r"):
            line1 = line1.strip()
            im_names.append(line1)
        im_names = np.array(im_names)
        # print(im_names)

        for line2 in open("./data/refnames.txt", "r"):
            line2 = line2.strip()
            ref_names.append(line2)
        ref_names = np.array(ref_names)
        # print(ref_names)

        self.patches1 = ()

        self.label = []

        self.refpatches = ()

        # im_names = [Info[Info['im_names'][0, :][i]].value.tobytes() \
        #                 [::2].decode() for i in self.index]
        # ref_names = [Info[Info['ref_names'][0, :][i]][()].tobytes()[::2].decode()
        #              for i in (ref_ids[self.index] - 1).astype(int)]
        im_names = [im_names[i] for i in self.index]
        ref_names = [ref_names[i] for i in self.index]

        self.mos = [self.mos[i] for i in self.index]

        for idx in range(len(self.index)):
            # print("Preprocessing Image: {} {} {}".format(im_names[idx], self.mos[idx], ref_names[idx]))
            im = self.loader(os.path.join(im_dir, im_names[idx]))
            refimg = self.loader(os.path.join(im_ref, ref_names[idx]))

            # patches1 = OverlappingCropPatches(im, self.patch_size,self.stride)  ################
            # refpatches = OverlappingCropPatches(refimg, self.patch_size, self.stride)
            if status == 'train':
                patches1 = OverlappingCropPatches(im, self.patch_size, self.stride)
                refpatches = OverlappingCropPatches(refimg, self.patch_size, self.stride)

                self.patches1 = self.patches1 + patches1
                self.refpatches = self.refpatches + refpatches

                for i in range(len(patches1)):
                    self.label.append(self.mos[idx])

            else:
                patches1 = OverlappingCropPatches(im, 64, 64)
                refpatches = OverlappingCropPatches(refimg, 64, 64)

                self.patches1 = self.patches1 + (torch.stack(patches1),)
                self.refpatches = self.refpatches + (torch.stack(refpatches),)
                self.label.append(self.mos[idx])

    def __len__(self):
        return len(self.patches1)

    def __getitem__(self, idx):

        return self.patches1[idx], (torch.Tensor([self.label[idx], ]), self.refpatches[idx])

import os
import torch
import argparse
import random
import numpy as np
from dataset_folders import pil_loader
from IQASolver import demoIQASolver
import torchvision
from IQAnetwork import *

torch.cuda.set_device(0)

def main(config):
    if config.mode == "train":
        folder_path = {
            'live': 'H:/0datasets/0datasets/LIVE/',
            'csiq': 'H:/0datasets/0datasets/CSIQ/',
            'tid2013': 'H:/0datasets/0datasets/tid2013/',
            'livec': 'H:/0datasets/0datasets/LIVEC/',
            'koniq-10k': 'H:/0datasets/0datasets/KonIQ-10k/',
            'cid2013': 'H:/0datasets/0datasets/CID2013/',
        }
        img_num = {
            'live': list(range(0, 29)),
            'csiq': list(range(0, 30)),
            'tid2013': list(range(0, 25)),
            'livec': list(range(0, 1162)),
            'koniq-10k': list(range(0, 10073)),
            'cid2013': list(range(0, 6)),
        }
        sel_num = img_num[config.dataset]
        srcc_all = np.zeros(config.train_test_num, dtype=np.float)
        plcc_all = np.zeros(config.train_test_num, dtype=np.float)
        print('Training and testing on %s dataset for %d rounds...' % (config.dataset, config.train_test_num))
        for i in range(config.train_test_num):
            print('Round %d' % (i + 1))
            # Randomly select 80% images for training and the rest for testing
            random.shuffle(sel_num)
            train_index = sel_num[0:int(round(0.8 * len(sel_num)))]
            test_index = sel_num[int(round(0.8 * len(sel_num))):len(sel_num)]
            solver = demoIQASolver(config, folder_path[config.dataset], train_index, test_index)
            srcc_all[i], plcc_all[i] = solver.train()
        srcc_med = np.mean(srcc_all)
        plcc_med = np.mean(plcc_all)
        print('Testing mean SRCC %4.4f,\tmedian PLCC %4.4f' % (srcc_med, plcc_med))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', dest='mode', type=str, default='train', help='train')
    parser.add_argument('--dataset', dest='dataset', type=str, default='live', help='Support datasets: livec|koniq-10k|cid2013|live|csiq|tid2013')
    parser.add_argument('--train_patch_num', dest='train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', dest='test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    parser.add_argument('--lr', dest='lr', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--weight_decay', dest='weight_decay', type=float, default=0, help='Weight decay')
    parser.add_argument('--lr_ratio', dest='lr_ratio', type=int, default=10, help='Learning rate ratio for hyper network')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', dest='epochs', type=int, default=16, help='Epochs for training')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_test_num', dest='train_test_num', type=int, default=10, help='Train-test times')

    config = parser.parse_args()
    main(config)

# CACIOUS CODING
# Data     : 6/1/23  8:55 PM
# File name: opt
# Desc     :
import argparse


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reid_dim', default=128, type=int, help='mot')
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--flip_odds', default=0.5, type=float, help='odds of flip of augmentation')
    parser.add_argument('--sv_augmentation', default=True, type=bool)
    parser.add_argument('--head_conv', type=int, default=-1,
                        help='conv layer channels for output head'
                             '0 for no conv layer'
                             '-1 for default setting: '
                             '256 for resnets and 256 for dla.')
    parser.add_argument('--load_model', default='weight/dla_training_weights_20.pth', help='path to pretrained model')
    parser.add_argument('--dataset_path',
                        default="../dataset/reid_dataset_0724",
                        help='path to training dataset')
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--IDn', default=67, type=int)
    parser.add_argument('--arch', default='dla_34',
                        help='model architecture. Currently tested'
                             'resdcn_34 | resdcn_50 | resfpndcn_34 |'
                             'dla_34 | hrnet_18')
    parser.add_argument('--gpus', default='0',
                        help='-1 for CPU, use comma for multiple gpus')
    opt = parser.parse_args()
    opt.gpus_str = opt.gpus
    opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]

    # 只输出reid
    opt.heads = {'id': opt.reid_dim}

    # init default head_conv
    if opt.head_conv == -1:
        opt.head_conv = 256 if 'dla' in opt.arch else 256

    opt.img_size = (128, 128)
    opt.down_ratio = 4

    return opt

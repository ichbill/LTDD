import os
import sys
sys.path.append("../")
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, epoch, evaluate_baseline, reduce_dataset, reduce_dataset_random, evaluate_balance
import wandb
import copy
import random
from reparam_module import ReparamModule
from torchvision import datasets, transforms
import warnings
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore", category=DeprecationWarning)



def main(args):

    
    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args,baseline=True)
    
    # dst_train = reduce_dataset(dst_train, rate=args.ipc/500, class_num=num_classes, num_per_class = 500)
    if args.random:
        first_stage_train = reduce_dataset_random(dst_train, class_num=num_classes, num_per_class = 50)
    else:
        first_stage_train = dst_train

    train_loader = DataLoader(
            first_stage_train, batch_size=args.batch_train, shuffle=True, num_workers=2
        )

    from collections import defaultdict
    class_counts = defaultdict(int)

    for i, datum in enumerate(train_loader):
        images, labels = datum
        for label in labels:
            class_counts[label.item()] += 1

    print(class_counts)

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None


    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    args.distributed = torch.cuda.device_count() > 1

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)
    

    criterion = nn.CrossEntropyLoss().to(args.device)



    args.lr_net = torch.tensor(args.lr_teacher).to(args.device)



    for model_eval in model_eval_pool:
        print('Evaluating: '+model_eval)
        network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        net, acc_train, acc_test, _ = evaluate_baseline(0, copy.deepcopy(network), train_loader, testloader, args, num_classes, texture=False)
        torch.save(net.state_dict(), os.path.join(args.save_dir, model_eval+'.pth'))

    print("Second stage evaluation")
    criterion = nn.CrossEntropyLoss().to(args.device)
    for model_eval in model_eval_pool:
        print('Evaluating: '+model_eval)
        network = get_network(model_eval, channel, num_classes, im_size, dist=False).to(args.device)
        network.load_state_dict(torch.load(os.path.join(args.save_dir, model_eval+'.pth')))

        active_layers = [network.classifier.weight, network.classifier.bias]
        for param in network.parameters():
            param.requires_grad = False
        for param in active_layers:
            param.requires_grad = True

        net, acc_train, acc_test, acc_class = evaluate_balance(0, copy.deepcopy(network), dst_train, testloader, args, num_classes, texture=False, train_criterion=criterion, hard_label=True)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=128, help='batch size for training networks')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='../dataset', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')
    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')
    parser.add_argument('--data_dir', type=str, default='path', help='dataset')
    parser.add_argument('--label_dir', type=str, default='path', help='dataset')
    parser.add_argument('--lr_dir', type=str, default='path', help='dataset')
    parser.add_argument('--parall_eva', type=bool, default=False, help='dataset')
    parser.add_argument('--ASL_model', type=str, default=None)
    parser.add_argument('--ASL_model_dir', type=str, default=None)
    parser.add_argument('--method', type=str, default='')

    parser.add_argument("--imbalance_rate", type=float, default=0.005)
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--save_dir", type=str, help="first_stage save dir")
    parser.add_argument('--epoch_eval_second', type=int, default=10, help='epochs to train a model on second stage')
    

    args = parser.parse_args()
    main(args)



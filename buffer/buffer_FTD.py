import os
import argparse
import sys 
sys.path.append("../")
import torch
import torch.nn as nn
from tqdm import tqdm
from utils.utils_gsam import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug
import copy

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import logging
from utils.weight_balancing.class_balanced_loss import CB_loss
from utils import utils_baseline
from utils.class_aware_sampler import get_sampler
from collections import defaultdict
import yaml
import time
import random
import torch.nn.functional as F
# from utils.dataset_cifar10imbalance import Cifar10Imbanlance
# from utils.dataset_cifar100imbalance import Cifar100Imbanlance

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.weight = weight

    def forward(self, inputs, targets):
        # Compute log softmax
        log_probs = F.log_softmax(inputs, dim=1)

        # Gather the log probabilities by using targets as indices
        loss = F.nll_loss(log_probs, targets, reduction='none')

        # Apply weights
        if self.weight is not None:
            weights = self.weight[targets]
            loss = loss * weights

        # Apply reduction
        if self.reduction == 'none':
            return loss
        elif self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")  
        

def balanced_softmax_loss(labels, logits, sample_per_class, reduction):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      sample_per_class: A int tensor of size [no of classes].
      reduction: string. One of "none", "mean", "sum"
    Returns:
      loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss


def main(args):
    with open(args.cfg, "r") as ymlfile:
        configs = yaml.safe_load(ymlfile)
    for key, value in configs.items():
        if hasattr(args, key):
            if getattr(args, key) == parser.get_default(key):
                setattr(args, key, value)
        else:
            setattr(args, key, value)    

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    # stage = ""
    # if args.first_stage_expert:
    #     stage = "first_stage"
    #     if args.l2 != 0:
    #         stage += "_wb"
    #     if args.first_stage_cutmix:
    #         stage += "_cutmix"
    # elif args.second_stage_expert:
    #     stage = "second_stage"
    #     if args.second_stage_maxnorm:
    #         stage += "_maxnorm"
    # log_file_path = os.path.join(args.buffer_path, f"{args.dataset}_{args.model}_{stage}.txt")
    # if args.test:
    #     log_file_path = os.path.join(args.buffer_path, f"{args.dataset}_{args.model}_{stage}_test.txt")
    log_file_path = args.log_file_path
    if not os.path.exists(os.path.dirname(log_file_path)):
        os.makedirs(os.path.dirname(log_file_path))

    logging.basicConfig(level=logging.INFO,
                        filename=log_file_path,
                        filemode='a',
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    logger = logging.getLogger(__name__)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)  # Set the log level for the console
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info('Loading data...')
    time_data_start = time.time()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    logger.info('Data loaded in {} seconds'.format(time.time() - time_data_start))
    # logger.info('\n================== Exp %d ==================\n '%exp)
    # breakpoint()
    # logger.info('Hyper-parameters: \n', args.__dict__)
    # logger.info('Hyper-parameters: \n{}'.format(args.__dict__))
    logger.info('Hyper-parameters:')
    for key, value in args.__dict__.items():
        logger.info(f'{key}: {value}')
    logger.info('\n')

    # save_dir = os.path.join(args.buffer_path, args.dataset)
    # if args.dataset == "ImageNet":
    #     save_dir = os.path.join(save_dir, args.subset, str(args.res))
    # if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
    #     save_dir += "_NO_ZCA"
    # if args.first_stage_expert or args.second_stage_expert:
    #     save_dir = os.path.join(save_dir, stage)
    # else:
    #     save_dir = os.path.join(save_dir, args.model)
    # if args.save_dir is not None:
    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    logger.info("BUILDING DATASET")
    for i in tqdm(range(len(dst_train))):
        sample = dst_train[i]
        if sample[0].shape[0] != 3:
            continue
            # breakpoint()
            # sample_image = sample[0].repeat(3, 1, 1)
        else:
            sample_image = sample[0]
        images_all.append(torch.unsqueeze(sample_image, dim=0))
        if args.dataset == 'ImageNet_LT':
            labels_all.append(sample[1])
        else:
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
    #logger.info('num of training images',len(images_all))
    len_dst_train = len(images_all)  ##50000

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)
    images_all = torch.cat(images_all, dim=0).to("cpu")
    labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    for c in range(num_classes):
        logger.info('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        logger.info('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    trajectories = []

    dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    if args.second_stage_expert or args.expert_oversampling:
        sampler_dic = {'sampler':get_sampler(),
                       'params':{'num_samples_cls':4}}
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=False, num_workers=0, sampler=sampler_dic['sampler'](dst_train, **sampler_dic['params']))
    else:
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

    class_counts = defaultdict(int)
    for i, datum in enumerate(trainloader):
        images, labels = datum
        for label in labels:
            class_counts[label.item()] += 1
    print("Class counts for training: ", class_counts)

    criterion = nn.CrossEntropyLoss().to(args.device)
    
    samples_per_cls = [0] * (max(class_counts.keys()) + 1)  # Initialize list with zeros
    for key, value in class_counts.items():
        samples_per_cls[key] = value
    if args.class_balanced_loss:
        def CB_lossFunc(logits, labelList): #defince CB loss function
            return CB_loss(labelList, logits, samples_per_cls, num_classes, "softmax", 0.9999, 2.0, args.device)
        criterion = CB_lossFunc

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    logger.info('DC augmentation parameters: \n', args.dc_aug_param)

    for it in range(0, args.num_experts):

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model
        teacher_net.train()
        if args.second_stage_expert:
            ckpt_path = os.path.join(args.ckpt_dir, "ckpt_{}.pt".format(it))
            teacher_net.load_state_dict(torch.load(ckpt_path))
            active_layers = [teacher_net.classifier.weight, teacher_net.classifier.bias]
            for param in teacher_net.parameters():
                param.requires_grad = False
            for param in active_layers:
                param.requires_grad = True
        lr = args.lr_teacher
       
       
        ##modification: using FTD here 
        from gsam import GSAM, LinearScheduler, CosineScheduler, ProportionScheduler
        base_optimizer = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=float(args.l2))
        # scheduler = CosineScheduler(T_max=args.train_epochs*len_dst_train, max_value=lr, min_value=0.0, 
            # optimizer=base_optimizer)
        if args.GSAM:
            scheduler = torch.optim.lr_scheduler.StepLR(base_optimizer,step_size=args.train_epochs*len(trainloader),gamma=1)
            rho_scheduler = ProportionScheduler(pytorch_lr_scheduler=scheduler, max_lr=lr, min_lr=lr,
                max_value=args.rho_max, min_value=args.rho_min)
            teacher_optim = GSAM(params=teacher_net.parameters(), base_optimizer=base_optimizer, 
                    model=teacher_net, gsam_alpha=args.alpha, rho_scheduler=rho_scheduler, adaptive=args.adaptive)
        else:
            teacher_optim = base_optimizer
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]
        for e in range(args.train_epochs):
            if args.GSAM:
                train_loss, train_acc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                            criterion=criterion, args=args, aug=True,scheduler=scheduler)

                test_loss, test_acc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                            criterion=criterion, args=args, aug=False, scheduler=scheduler)
            else:
                train_loss, train_acc, _ = utils_baseline.epoch('train', trainloader, teacher_net, teacher_optim, criterion, args, num_classes, aug=True, hard_label=True)
                test_loss, test_acc, _ = utils_baseline.epoch('test', testloader, teacher_net, None, criterion, args, num_classes, aug=False, If_Float=False)

                if e in lr_schedule:
                    lr *= 0.1
                    teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

            logger.info("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTest Acc: {}".format(it, e, train_acc, test_acc))

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        if args.first_stage_expert:
            ckpt_dir = os.path.join(save_dir, "ckpt")
            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            ckpt_path = os.path.join(ckpt_dir, "ckpt_{}.pt".format(it))
            logger.info("Saving {}".format(ckpt_path))
            torch.save(teacher_net.state_dict(), ckpt_path)

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            logger.info("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []


if __name__ == '__main__':
    random.seed(42)

    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--cfg', type=str, default='config.yaml', help='config file')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    #parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument("--rho_max", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--rho_min", default=2.0, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--alpha", default=0.4, type=float, help="Rho parameter for SAM.")
    parser.add_argument("--adaptive", default=True, type=bool, help="True if you want to use the Adaptive SAM.")
    parser.add_argument("--imbalance_rate", type=float, default=0.005)
    # parser.add_argument("--first_stage", action='store_true', help="The first stage of the decoupling training.")
    # parser.add_argument("--second_stage", action='store_true', help="The second stage of the decoupling training.")

    args = parser.parse_args()
    time_start = time.time()
    main(args)
    time_end = time.time()
    print('time cost', time_end-time_start, 's')
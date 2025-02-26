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
from utils.utils_baseline import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from reparam_module import ReparamModule
# from kmeans_pytorch import kmeans
from utils.cfg import CFG as cfg
import warnings
import yaml
import time
from utils import utils_two_stage
import datetime

warnings.filterwarnings("ignore", category=DeprecationWarning)

def manual_seed(seed=0):
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

def main(args):

    manual_seed()

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(x) for x in args.device])

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.skip_first_eva==False:
        eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    else:
        eval_it_pool = np.arange(args.eval_it, args.Iteration + 1, args.eval_it).tolist()
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
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

    args.distributed = torch.cuda.device_count() > 1
    cur_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    wandb_name = 'imbrate_'+str(args.imbalance_rate)+'_match_both_fl_'+str(args.first_stage_lambda)+'_sl_'+str(args.second_stage_lambda)+'_distributed_'+str(args.distributed)+"_"+cur_time

    wandb.init(sync_tensorboard=False,
               project=args.project,
               job_type="CleanRepo",
               config=args,
               name=wandb_name,
               )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    images_all = []
    labels_all = []
    indices_class = [[] for c in range(num_classes)]
    print("BUILDING DATASET")
    if args.dataset == 'ImageNet1K' and os.path.exists('images_all.pt') and os.path.exists('labels_all.pt'):
        images_all = torch.load('images_all.pt')
        labels_all = torch.load('labels_all.pt')
    else:
        for i in tqdm(range(len(dst_train))):
            sample = dst_train[i]
            images_all.append(torch.unsqueeze(sample[0], dim=0))
            labels_all.append(class_map[torch.tensor(sample[1]).item()])
        images_all = torch.cat(images_all, dim=0).to("cpu")
        labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")
        if args.dataset == 'ImageNet1K':
            torch.save(images_all, 'images_all.pt')
            torch.save(labels_all, 'labels_all.pt')

    for i, lab in tqdm(enumerate(labels_all)):
        indices_class[lab].append(i)



    for c in range(num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    for ch in range(channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    def get_real_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        if len(idx_shuffle) < n:
            if args.duplicate:
                # idx_shuffle = np.random.choice(indices_class[c], n, replace=True)
                num_additional_images = n - len(idx_shuffle)
                print("Selected {} images, duplicating {} images".format(len(idx_shuffle), num_additional_images))
                while num_additional_images > 0:
                    # print(f"current: {len(temp_img)}, need: {num_additional_images}", end=",")
                    print(f"current: {len(idx_shuffle)}, need: {num_additional_images}", end=",")
                    if num_additional_images >= len(idx_shuffle):
                        print(f"adding {len(idx_shuffle)}", end=",")
                        additional_images = idx_shuffle
                        num_additional_images -= len(idx_shuffle)
                    else:
                        print(f"adding {num_additional_images}", end=",")
                        additional_images = idx_shuffle[:num_additional_images]
                        num_additional_images = 0
                    idx_shuffle = np.concatenate((idx_shuffle, additional_images), axis=0)
                    print(f"sum: {len(idx_shuffle)}")
        return images_all[idx_shuffle]


    ''' initialize the synthetic data '''
    label_syn = torch.tensor([ [i] * args.ipc for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]


    image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.first_stage_expert_dir is not None:
        first_stage_expert_dir = args.first_stage_expert_dir
        print("First Stage Expert Dir: {}".format(first_stage_expert_dir))
        if args.load_all:
            fs_buffer = utils_two_stage.load_buffer(first_stage_expert_dir, args)
            ss_buffer = fs_buffer
        else:
            fs_buffer, fs_buffer_id, fs_file_idx, fs_expert_idx, fs_expert_files, fs_expert_id = utils_two_stage.load_buffer(first_stage_expert_dir, args)
            ss_buffer, ss_buffer_id, ss_file_idx, ss_expert_idx, ss_expert_files, ss_expert_id =  fs_buffer, fs_buffer_id, fs_file_idx, fs_expert_idx, fs_expert_files, fs_expert_id

    if args.second_stage_expert_dir is not None:
        second_stage_expert_dir = args.second_stage_expert_dir
        print("Second Stage Expert Dir: {}".format(second_stage_expert_dir)) 
        if args.load_all:
            ss_buffer = utils_two_stage.load_buffer(second_stage_expert_dir, args)
        else:
            ss_buffer, ss_buffer_id, ss_file_idx, ss_expert_idx, ss_expert_files, ss_expert_id = utils_two_stage.load_buffer(second_stage_expert_dir, args)

    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(num_classes):
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_real_images(c, args.ipc).detach().data
            
    elif args.pix_init == 'samples_predicted_correctly':
        if args.parall_eva==False:
            device = torch.device("cuda:0")
        else:
            device = args.device
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed and args.parall_eva==True:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        label_expert_files = ss_expert_files
        temp_params = torch.load(label_expert_files[0])[0][args.Label_Model_Timestamp]
        temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0)
        if args.distributed and args.parall_eva==True:
            temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        for c in range(num_classes):
            data_for_class_c = get_images(c, len(indices_class[c])).detach().data
            n, ch, w, h = data_for_class_c.shape
            selected_num = 0
            select_times = 0
            cur=0
            temp_img = None
            Wrong_Predicted_Img = None
            batch_size = 256
            index = []
            index_start = 0
            if args.longtailipc and args.imbalance_rate==0.005 and args.ipc==50:
                longtailipc_list = [224,214,69,39,22,12,7,4,3,2]
                loopiter = longtailipc_list[c]
            else:
                loopiter = args.ipc
            while len(index)<loopiter:
                print(str(c)+'.'+str(select_times)+'.'+str(cur))
                current_data_batch = data_for_class_c[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
                if batch_size*select_times > len(data_for_class_c):
                    select_times = 0
                    cur+=1
                    temp_params = torch.load(label_expert_files[int(cur/10)%10])[cur%10][args.Label_Model_Timestamp]
                    temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0).to(device)
                    if args.distributed and args.parall_eva==True:
                        temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    continue
                logits = Temp_net(current_data_batch, flat_param=temp_params).detach()
                prediction_class = np.argmax(logits.cpu().data.numpy(), axis=-1)
                for i in range(len(prediction_class)):
                    if prediction_class[i]==c and len(index)<loopiter:
                        index.append(batch_size*select_times+i)
                        index=list(set(index))
                select_times+=1
                # if len(index) >= n:
                print(len(index))
                if (batch_size*select_times > n) or args.longtailipc:
                # if batch_size*select_times > n:
                    if len(index) > 0:
                        temp_img = torch.index_select(data_for_class_c, dim=0, index=torch.tensor(index))
                        num_images, channels, height, width = temp_img.shape
                        num_additional_images = args.ipc - num_images
                    else:
                        num_images = 0
                        num_additional_images = args.ipc
                        print("No correct predictions, generating {} noise images".format(num_additional_images))
                        temp_img = torch.randn(num_additional_images, ch, h, w, requires_grad=True)
                        break
                    if args.noise:
                        print("Selected {} images, generating {} noise images".format(num_images, num_additional_images))
                        additional_images = torch.randn(num_additional_images, channels, height, width, requires_grad=True)
                        temp_img = torch.cat((temp_img, additional_images), dim=0)
                    elif args.duplicate:
                        print("Selected {} images, duplicating {} images".format(num_images, num_additional_images))
                        while num_additional_images > 0:
                            print(f"current: {len(temp_img)}, need: {num_additional_images}", end=",")
                            if num_additional_images >= len(temp_img):
                                print(f"adding {len(temp_img)}", end=",")
                                additional_images = temp_img
                                num_additional_images -= len(temp_img)
                            else:
                                print(f"adding {num_additional_images}", end=",")
                                additional_images = temp_img[:num_additional_images]
                                num_additional_images = 0
                            temp_img = torch.cat((temp_img, additional_images), dim=0)
                            print(f"sum: {len(temp_img)}") 
                    elif args.inherent:
                        print("Selected {} images, using inherent strategy.".format(num_images))
                        break
                    elif args.longtailipc:
                        # print("Selected {} images, generating {} noise images for longtail distribution".format(num_images, num_additional_images))
                        # additional_images = torch.randn(num_additional_images, channels, height, width, requires_grad=True)
                        # temp_img = torch.cat((temp_img, additional_images), dim=0)
                        print("Selected {} images, using longtailipc strategy.".format(num_images))
                        break
                    break
                if len(index) == args.ipc:
                    temp_img = torch.index_select(data_for_class_c, dim=0, index=torch.tensor(index))
                    break
            if args.inherent:
                image_syn.data[index_start:index_start+len(temp_img)] = temp_img.detach()
                index_start += len(temp_img)
            elif args.longtailipc and args.imbalance_rate==0.005 and args.ipc==50:
                # longtailipc_list = [224,214,69,39,22,12,7,4,3,2]
                numsamples = longtailipc_list[c]
                image_syn.data[index_start:index_start+numsamples] = temp_img[:numsamples].detach()
                index_start += numsamples
            else:
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = temp_img.detach()
    else:
        print('initialize synthetic data from random noise')


    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=0.5)


    
    optimizer_img.zero_grad()

    ###

    '''test'''
    def SoftCrossEntropy(inputs, target, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(input_log_likelihood, target_log_likelihood)) / batch
        return loss
    
    def Balanced_softmax_loss(logits, labels, sample_per_class, reduction='average', alpha=1.):
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
        logits = logits - spc.log()
        # sample_per_class = sample_per_class / sample_per_class.sum()
        loss = WeightedSoftCrossEntropy(inputs=logits, target=labels, weights=sample_per_class, reduction=reduction)
        # loss = SoftCrossEntropy(inputs=logits, target=labels, reduction=reduction)
        return loss
    
    def WeightedSoftCrossEntropy(inputs, target, weights, reduction='average'):
        input_log_likelihood = -F.log_softmax(inputs, dim=1)
        target_log_likelihood = F.softmax(target, dim=1)
        batch = inputs.shape[0]
        loss = torch.sum(torch.mul(torch.mul(input_log_likelihood, target_log_likelihood), weights)) / batch
        return loss

    if args.hard_label:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = SoftCrossEntropy

    print('%s training begins'%get_time())
    best_acc = {m: 0 for m in model_eval_pool}
    best_std = {m: 0 for m in model_eval_pool}

    '''------test------'''
    '''only sum correct predicted logits'''
    if args.pix_init == "samples_predicted_correctly":
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        batch_size = 256
        for i in range(len(label_expert_files)):
            Temp_Buffer = torch.load(label_expert_files[i])
            for j in Temp_Buffer:
                temp_logits = None
                for select_times in range((len(image_syn)+batch_size-1)//batch_size):
                    current_data_batch = image_syn[batch_size*select_times : batch_size*(select_times+1)].detach().to(device)
                    Temp_params = j[args.Label_Model_Timestamp]
                    Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
                    if args.distributed:
                        Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    Initialized_Labels = Temp_net(current_data_batch, flat_param=Initialize_Labels_params)
                    if temp_logits == None:
                        temp_logits = Initialized_Labels.detach()
                    else:
                        temp_logits = torch.cat((temp_logits, Initialized_Labels.detach()),0)
                logits.append(temp_logits.detach().cpu())
        logits_tensor = torch.stack(logits)
        true_labels = label_syn.cpu()
        predicted_labels = torch.argmax(logits_tensor, dim=2).cpu()
        correct_predictions = predicted_labels == true_labels.view(1, -1)
        mask = correct_predictions.unsqueeze(2)
        correct_logits = logits_tensor * mask.float()
        correct_logits_per_model = correct_logits.sum(dim=0)
        num_correct_images_per_model = correct_predictions.sum(dim=0, dtype=torch.float)
        zero_count_mask = num_correct_images_per_model == 0
        num_correct_images_per_model[zero_count_mask] = 1e-4
        average_logits_per_image = correct_logits_per_model / num_correct_images_per_model.unsqueeze(1) 
        Initialized_Labels = average_logits_per_image

    elif args.pix_init == "real":
        Temp_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model
        Temp_net = ReparamModule(Temp_net)
        if args.distributed:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        Temp_params = ss_buffer[0][-1]
        Initialize_Labels_params = torch.cat([p.data.to(args.device).reshape(-1) for p in Temp_params], 0)
        if args.distributed:
            Initialize_Labels_params = Initialize_Labels_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        Initialized_Labels = Temp_net(image_syn, flat_param=Initialize_Labels_params)

    # assign lab_syn
    if not args.pix_init == "load_previous":
        acc = np.sum(np.equal(np.argmax(Initialized_Labels.cpu().data.numpy(), axis=-1), label_syn.cpu().data.numpy()))
        print('InitialAcc:{}'.format(acc/len(label_syn)))
        if not args.hard_label:
            label_syn = copy.deepcopy(Initialized_Labels.detach()).to(args.device).requires_grad_(True)
            label_syn.requires_grad=True
        label_syn = label_syn.to(args.device)

        del Temp_net
    else:
        label_syn = label_syn.detach().to(args.device).requires_grad_(True)  

    optimizer_y = torch.optim.SGD([label_syn], lr=args.lr_y, momentum=args.Momentum_y)
    vs = torch.zeros_like(label_syn)
    accumulated_grad = torch.zeros_like(label_syn)
    last_random = 0    

    # test
    curMax_times = 0
    current_accumulated_step = 0

    if args.weighted_loss or args.BFLoss:
        weights = [len(indices_class[c]) for c in range(num_classes)]
        if args.BFLoss:
            first_weights = [x/sum(weights) for x in weights]
        elif args.weighted_loss:
            # first_weights = [x/sum(weights) for x in weights]
            first_weights = [(x+args.first_weight_factor)/(sum(weights)+args.first_weight_factor) for x in weights]
        # weights = [(x/sum(weights))**0.5 for x in weights]
        # weights = [(x+sum(weights)/len(weights))/sum(weights) for x in weights]
        second_weights = [(x+args.second_weight_factor)/(sum(weights)+args.second_weight_factor) for x in weights]
        
        first_weights = torch.tensor(first_weights, dtype=torch.float, device=args.device)
        second_weights = torch.tensor(second_weights, dtype=torch.float, device=args.device)
        print("First Weights: ", first_weights)
        print("Second Weights: ", second_weights)
    expert_dict = {}
    expert_first_dict = {}
    expert_second_dict = {}
    expert_first_dict['expert_files'] = fs_expert_files
    expert_first_dict['expert_idx'] = fs_expert_idx
    expert_first_dict['expert_id'] = fs_expert_id
    expert_first_dict['buffer'] = fs_buffer
    expert_first_dict['buffer_id'] = fs_buffer_id
    expert_first_dict['file_idx'] = fs_file_idx
    expert_first_dict['stage_lambda'] = args.first_stage_lambda
    expert_first_dict['match_classifier'] = args.match_classifier
    expert_first_dict['weighted_loss'] = args.weighted_loss
    expert_first_dict['BFLoss'] = args.BFLoss
    expert_first_dict['weights'] = first_weights
    expert_first_dict['match_finetune'] = False # never match finetune in the first stage
    expert_first_dict['min_start_epoch'] = args.min_start_epoch
    expert_first_dict['current_max_start_epoch'] = args.current_max_start_epoch
    expert_first_dict['max_start_epoch'] = args.max_start_epoch
    expert_first_dict['syn_steps'] = args.syn_steps

    expert_second_dict['expert_files'] = ss_expert_files
    expert_second_dict['expert_idx'] = ss_expert_idx
    expert_second_dict['expert_id'] = ss_expert_id
    expert_second_dict['buffer'] = ss_buffer
    expert_second_dict['buffer_id'] = ss_buffer_id
    expert_second_dict['file_idx'] = ss_file_idx
    expert_second_dict['stage_lambda'] = args.second_stage_lambda
    expert_second_dict['match_classifier'] = True # always match classifier in the second stage
    expert_second_dict['weighted_loss'] = args.second_stage_weighted_loss
    expert_second_dict['BFLoss'] = False # never use BF loss in the second stage
    expert_second_dict['weights'] = second_weights
    expert_second_dict['match_finetune'] = True # always match finetune in the second stage
    expert_second_dict['min_start_epoch'] = args.second_min_start_epoch
    expert_second_dict['current_max_start_epoch'] = args.second_current_max_start_epoch
    expert_second_dict['max_start_epoch'] = args.second_max_start_epoch
    expert_second_dict['syn_steps'] = args.syn_steps_second

    expert_dict['first'] = expert_first_dict
    expert_dict['second'] = expert_second_dict

    if args.model == "ConvNet":
        classifier_start = 299520 
    elif args.model == "ResNet18":
        classifier_start = 11168832
    elif args.model == "ConvNetD4" and args.dataset == "Tiny_LT":
        classifier_start = 447360

    for it in range(0, args.Iteration+1):
        save_this_it = False
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                accs_test = []
                accs_train = []
                accs_class = []

                for it_eval in range(args.num_eval):
                    if args.parall_eva==False:
                        device = torch.device("cuda:0")
                        net_eval = get_network(model_eval, channel, num_classes, im_size, dist=False).to(device) # get a random model
                    else:
                        device = args.device
                        net_eval = get_network(model_eval, channel, num_classes, im_size, dist=True).to(device) # get a random model

                    eval_labs = label_syn.detach().to(device)
                    with torch.no_grad():
                        image_save = image_syn.to(device)
                    image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()).to(device), copy.deepcopy(eval_labs.detach()).to(device) # avoid any unaware modification

                    args.lr_net = syn_lr.item()
                    _, acc_train, acc_test, acc_class = evaluate_synset(it_eval, copy.deepcopy(net_eval).to(device), image_syn_eval.to(device), label_syn_eval.to(device), testloader, args, num_classes, Epoch=int(args.epoch_eval_train), texture=False, train_criterion=criterion)
                    accs_test.append(acc_test)
                    accs_train.append(acc_train)
                    if args.dataset == "CIFAR10_LT":
                        accs_class.append(acc_class)

                accs_test = np.array(accs_test)
                accs_train = np.array(accs_train)
                acc_test_mean = np.mean(accs_test)
                acc_test_std = np.std(accs_test)
                if args.dataset == "CIFAR10_LT":
                    accs_class = np.array(accs_class)
                    acc_class_mean = np.mean(accs_class, axis=0)
                    acc_class_mean_log = {f"Class_acc_{j}": acc_class_mean[j] for j in range(acc_class_mean.shape[0])}

                if acc_test_mean > best_acc[model_eval]:
                    best_acc[model_eval] = acc_test_mean
                    best_std[model_eval] = acc_test_std
                    save_this_it = True
                print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, acc_test_mean, acc_test_std))
                wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                wandb.log({'Std/{}'.format(model_eval): acc_test_std}, step=it)
                wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)
                if args.dataset == "CIFAR10_LT":
                    wandb.log(acc_class_mean_log, step=it)

        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()
                save_dir = os.path.join(".", "logged_files", args.dataset, str(args.ipc), args.model, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(os.path.join(save_dir,'Normal'))
                    
                torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal',"images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_{}.pt".format(it)))
                torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, 'Normal', "labels_best.pt".format(it)))
                    torch.save(syn_lr.detach().cpu(), os.path.join(save_dir, 'Normal', "lr_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()
                        torch.save(image_save.cpu(), os.path.join(save_dir, 'Normal', "images_zca_{}.pt".format(it)))
                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)
                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        accum_image_grad = torch.zeros_like(image_syn)
        accum_label_grad = torch.zeros_like(label_syn)
        accum_lr_grad = torch.zeros_like(syn_lr)
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        optimizer_y.zero_grad()
        for phase in ['first', 'second']:
            expert_files = expert_dict[phase]['expert_files']
            expert_idx = expert_dict[phase]['expert_idx']
            expert_id = expert_dict[phase]['expert_id']
            buffer = expert_dict[phase]['buffer']
            buffer_id = expert_dict[phase]['buffer_id']
            file_idx = expert_dict[phase]['file_idx']
            stage_lambda = expert_dict[phase]['stage_lambda']
            match_classifier = expert_dict[phase]['match_classifier']
            weighted_loss = expert_dict[phase]['weighted_loss']
            BFLoss = expert_dict[phase]['BFLoss']
            weights = expert_dict[phase]['weights']
            match_finetune = expert_dict[phase]['match_finetune']
            min_start_epoch = expert_dict[phase]['min_start_epoch']
            current_max_start_epoch = expert_dict[phase]['current_max_start_epoch']
            max_start_epoch = expert_dict[phase]['max_start_epoch']
            syn_steps = expert_dict[phase]['syn_steps']

            wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

            student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

            student_net = ReparamModule(student_net)

            if args.distributed:
                student_net = torch.nn.DataParallel(student_net)

            student_net.train()

            num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

            if args.load_all:
                expert_trajectory = buffer[np.random.randint(0, len(buffer))]
            else:
                expert_trajectory = buffer[buffer_id[expert_idx]]
                expert_idx += 1
                if expert_idx == len(buffer):
                    expert_idx = 0
                    file_idx += 1
                    if file_idx == len(expert_files):
                        file_idx = 0
                        random.shuffle(expert_id)
                    print("loading file {}".format(expert_files[expert_id[file_idx]]))
                    if args.max_files != 1:
                        del buffer
                        buffer = torch.load(expert_files[expert_id[file_idx]])
                    if args.max_experts is not None:
                        buffer = buffer[:args.max_experts]
                    random.shuffle(buffer_id)

            expert_dict[phase]['expert_files'] = expert_files
            expert_dict[phase]['expert_idx'] = expert_idx
            expert_dict[phase]['expert_id'] = expert_id
            expert_dict[phase]['buffer'] = buffer
            expert_dict[phase]['buffer_id'] = buffer_id
            expert_dict[phase]['file_idx'] = file_idx

            # Only match easy traj. in the early stage
            if args.Sequential_Generation:
                Upper_Bound = current_max_start_epoch + int((max_start_epoch-current_max_start_epoch) * it/(args.expansion_end_epoch))
                Upper_Bound = min(Upper_Bound, max_start_epoch)
            else:
                Upper_Bound = max_start_epoch

            start_epoch = np.random.randint(min_start_epoch, Upper_Bound)

            starting_params = expert_trajectory[start_epoch]
            target_params = expert_trajectory[start_epoch+args.expert_epochs]

            if not match_classifier:
                target_params = target_params[:-2]
            target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)
            student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]
            starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)
            if not match_classifier:
                param_dist = torch.nn.functional.mse_loss(starting_params[:classifier_start], target_params, reduction="sum")
            else:
                param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")    
            syn_images = image_syn
            y_hat = label_syn

            syn_image_gradients = torch.zeros(image_syn.shape).to(args.device)
            syn_label_gradients = torch.zeros(label_syn.shape).to(args.device)
            x_list = []
            original_x_list = []
            y_list = []
            original_y_list = []
            indices_chunks = []
            gradient_sum = torch.zeros(student_params[-1].shape).to(args.device)
            indices_chunks_copy = []      
            
            for step in range(syn_steps):
                if not indices_chunks:
                    indices = torch.randperm(len(syn_images))
                    indices_chunks = list(torch.split(indices, args.batch_syn))

                these_indices = indices_chunks.pop()
                indices_chunks_copy.append(these_indices)

                x = syn_images[these_indices]
                this_y = y_hat[these_indices]
                original_x_list.append(x)
                original_y_list.append(this_y)
                if args.dsa and (not args.no_aug):
                    x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)
                x_list.append(x.clone())
                y_list.append(this_y.clone())

                if args.distributed and step == 0:
                    forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
                    forward_sum = gradient_sum.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                else:
                    forward_params = student_params[-1]
                    forward_sum = gradient_sum
                x = student_net(x, flat_param=forward_params)
                if weighted_loss or BFLoss:
                    if weighted_loss:
                        ce_loss = WeightedSoftCrossEntropy(x, this_y, weights)
                    elif BFLoss:
                        ce_loss = Balanced_softmax_loss(x, this_y, weights)            
                else:
                    ce_loss = criterion(x, this_y)

                grad = torch.autograd.grad(ce_loss, forward_params, create_graph=True, retain_graph=True)[0]
                if match_finetune:
                    if args.distributed:
                        grad[:,:classifier_start] = 0
                    else:
                        grad[:classifier_start] = 0
                # breakpoint()
                detached_grad = grad.detach().clone()
                student_params.append(student_params[-1] - syn_lr.item() * detached_grad)
                gradient_sum = detached_grad + forward_sum

                del grad

            # --------Compute the gradients regarding input image and learning rate---------
            # compute gradients invoving 2 gradients
            for i in range(syn_steps):
                # compute gradients for w_i
                w_i = student_params[i]
                if args.distributed and i == 0:
                    w_i = w_i.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                # breakpoint()
                output_i = student_net(x_list[i], flat_param = w_i)
                if args.batch_syn:
                    if weighted_loss:
                        ce_loss_i = WeightedSoftCrossEntropy(output_i, y_list[i], weights)
                    elif BFLoss:
                        ce_loss_i = Balanced_softmax_loss(output_i, y_list[i], weights)
                    else:
                        ce_loss_i = criterion(output_i, y_list[i])
                else:
                    if weighted_loss:
                        ce_loss_i = WeightedSoftCrossEntropy(output_i, y_hat, weights)
                    elif BFLoss:
                        ce_loss_i = Balanced_softmax_loss(output_i, y_hat, weights)
                    else:
                        ce_loss_i = criterion(output_i, y_hat)

                grad_i = torch.autograd.grad(ce_loss_i, w_i, create_graph=True, retain_graph=True)[0]
                # breakpoint()
                if not match_classifier:
                    if args.distributed:
                        single_term = syn_lr.item() * (target_params[:classifier_start] - starting_params[:classifier_start])
                        single_term = single_term.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                        square_term = (syn_lr.item() ** 2) * gradient_sum[:,:classifier_start]
                        total_term = 2 * (single_term + square_term) @ grad_i[:,:classifier_start].T / param_dist
                    else:
                        single_term = syn_lr.item() * (target_params[:classifier_start] - starting_params[:classifier_start])
                        square_term = (syn_lr.item() ** 2) * gradient_sum[:classifier_start]    
                        total_term = 2 * (single_term + square_term) @ grad_i[:classifier_start] / param_dist
                else:
                    single_term = syn_lr.item() * (target_params - starting_params)
                    square_term = (syn_lr.item() ** 2) * gradient_sum
                    if args.distributed:
                        single_term = single_term.unsqueeze(0).expand(torch.cuda.device_count(), -1)
                        total_term = 2 * (single_term + square_term) @ grad_i.T / param_dist
                    else:
                        total_term = 2 * (single_term + square_term) @ grad_i / param_dist
                # breakpoint()

                if args.distributed:
                    # breakpoint()
                    total_term = total_term.diagonal().mean()

                gradients_x, gradients_y = torch.autograd.grad(total_term, [original_x_list[i], original_y_list[i]] )
                with torch.no_grad():
                    syn_image_gradients[indices_chunks_copy[i]] += gradients_x
                    syn_label_gradients[indices_chunks_copy[i]] += gradients_y
            # ---------end of computing input image gradients and learning rates--------------

            image_syn.grad = syn_image_gradients
            label_syn.grad = syn_label_gradients
            # breakpoint()
            if not match_classifier:
                if args.distributed:
                    gradient_sum = gradient_sum.mean(dim=0)
                grand_loss = starting_params[:classifier_start] - syn_lr * gradient_sum[:classifier_start] - target_params[:classifier_start]
            else:
                if args.distributed:
                    gradient_sum = gradient_sum.mean(dim=0)
                grand_loss = starting_params - syn_lr * gradient_sum - target_params

            grand_loss = grand_loss.dot(grand_loss) / param_dist

            lr_grad,  = torch.autograd.grad(grand_loss, syn_lr)
            syn_lr.grad = lr_grad

            accum_image_grad += stage_lambda * image_syn.grad
            accum_label_grad += stage_lambda * label_syn.grad
            accum_lr_grad += stage_lambda * lr_grad

            # if grand_loss<=args.threshold:
            #     optimizer_y.step()
            #     optimizer_img.step()
            #     optimizer_lr.step()
            # else:
            #     wandb.log({"falts": start_epoch}, step=it)

            if phase == 'first':
                first_grand_loss = grand_loss.detach().cpu()*stage_lambda
            else:
                second_grand_loss = grand_loss.detach().cpu()*stage_lambda

            wandb.log({f"{phase}_stage_start_Epoch": start_epoch}, step=it)
            wandb.log({f"{phase}_stage_Loss": grand_loss.detach().cpu()*stage_lambda}, step=it)

            for _ in student_params:
                del _ 

        image_syn.grad = accum_image_grad
        label_syn.grad = accum_label_grad
        syn_lr.grad = accum_lr_grad

        optimizer_y.step()
        optimizer_img.step()
        optimizer_lr.step()

        if it%10 == 0:
            # print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))
            print('%s iter = %04d, first_loss = %.4f, second_loss = %.4f' % (get_time(), it, first_grand_loss.item(), second_grand_loss.item()))
    wandb.finish()


if __name__ == '__main__':
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument("--cfg", type=str, default="")
    args = parser.parse_args()
    
    cfg.merge_from_file(args.cfg)
    for key, value in cfg.items():
        arg_name = '--' + key
        parser.add_argument(arg_name, type=type(value), default=value)
    args = parser.parse_args()
    time_start = time.time()
    main(args)
    time_end = time.time()
    print('Totally cost', time_end - time_start)
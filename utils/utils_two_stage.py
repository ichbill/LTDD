import os
import torch
import random
import numpy as np

from utils.utils_baseline import get_network
from reparam_module import ReparamModule

def load_buffer(expert_dir, args):
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        return buffer
    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        # random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]

        expert_id = [i for i in range(len(expert_files))]
        random.shuffle(expert_id)

        print("loading file {}".format(expert_files[expert_id[file_idx]]))
        buffer = torch.load(expert_files[expert_id[file_idx]])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        buffer_id = [i for i in range(len(buffer))]
        random.shuffle(buffer_id)
        return buffer, buffer_id, file_idx, expert_idx, expert_files, expert_id
    
def generate_syn_data(args, cfg, expert_files, dataset_vars, real_data_vars, syn_data_vars):

    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(real_data_vars["indices_class"][c])[:n]
        return real_data_vars["images_all"][idx_shuffle]
    
    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(dataset_vars['num_classes']):
            syn_data_vars["image_syn"].data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data

    elif args.pix_init == 'samples_predicted_correctly':
        if args.parall_eva==False:
            device = torch.device("cuda:0")
        else:
            device = args.device
        if cfg.Initialize_Label_With_Another_Model:
            Temp_net = get_network(args.Initialize_Label_Model, dataset_vars['channel'], dataset_vars['num_classes'], dataset_vars['im_size'], dist=False).to(device)  # get a random model
        else:
            Temp_net = get_network(args.model, dataset_vars['channel'], dataset_vars['num_classes'], dataset_vars['im_size'], dist=False).to(device)  # get a random model
        Temp_net.eval()
        Temp_net = ReparamModule(Temp_net)
        if args.distributed and args.parall_eva==True:
            Temp_net = torch.nn.DataParallel(Temp_net)
        Temp_net.eval()
        logits=[]
        label_expert_files = expert_files
        temp_params = torch.load(label_expert_files[0])[0][args.Label_Model_Timestamp]
        temp_params = torch.cat([p.data.to(device).reshape(-1) for p in temp_params], 0)
        if args.distributed and args.parall_eva==True:
            temp_params = temp_params.unsqueeze(0).expand(torch.cuda.device_count(), -1)
        for c in range(dataset_vars['num_classes']):
            data_for_class_c = get_images(c, len(real_data_vars["indices_class"][c])).detach().data
            n, ch, w, h = data_for_class_c.shape
            selected_num = 0
            select_times = 0
            cur=0
            temp_img = None
            Wrong_Predicted_Img = None
            batch_size = 256
            index = []
            while len(index)<args.ipc:
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
                    if prediction_class[i]==c and len(index)<args.ipc:
                        index.append(batch_size*select_times+i)
                        index=list(set(index))
                select_times+=1
                # if len(index) >= n:
                print(len(index))
                if batch_size*select_times > n:
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
                    break
                if len(index) == args.ipc:
                    temp_img = torch.index_select(data_for_class_c, dim=0, index=torch.tensor(index))
                    break
            syn_data_vars["image_syn"].data[c * args.ipc:(c + 1) * args.ipc] = temp_img.detach()
    else:
        print('initialize synthetic data from random noise')

    return syn_data_vars["image_syn"]
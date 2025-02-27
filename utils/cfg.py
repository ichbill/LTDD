from yacs.config import CfgNode as CN


def show_cfg(cfg):
    dump_cfg = CN()
    dump_cfg.EXPERIMENT = cfg.EXPERIMENT
    dump_cfg.DATASET = cfg.DATASET
    dump_cfg.DISTILLER = cfg.DISTILLER
    dump_cfg.SOLVER = cfg.SOLVER
    dump_cfg.LOG = cfg.LOG
    if cfg.DISTILLER.TYPE in cfg:
        dump_cfg.update({cfg.DISTILLER.TYPE: cfg.get(cfg.DISTILLER.TYPE)})
    print(log_msg("CONFIG:\n{}".format(dump_cfg.dump()), "INFO"))


CFG = CN()

# Configuration Settings

# dataset
CFG.dataset = 'CIFAR10'

# ImageNet subset. This only does anything when --dataset=ImageNet
CFG.subset = 'imagenette'

# model
CFG.model = 'ConvNet'

# image(s) per class
CFG.ipc = 1

# eval_mode, check utils.py for more info
CFG.eval_mode = 'S'

# how many networks to evaluate on
CFG.num_eval = 5

# how often to evaluate
CFG.eval_it = 100

# epochs to train a model with synthetic data
CFG.epoch_eval_train = 1000

# how many distillation steps to perform
CFG.Iteration = 5000

# Learning rates
CFG.lr_img = 1000.0  # learning rate for updating synthetic images
CFG.lr_teacher = 0.01  # initialization for synthetic learning rate
CFG.lr_init = 0.01  # how to init lr (alpha)

# Batch sizes
CFG.batch_real = 256  # batch size for real data
CFG.batch_syn = None  # should only use this if you run out of VRAM
CFG.batch_train = 256  # batch size for training networks

# Initialization for synthetic images
CFG.pix_init = 'samples_predicted_correctly'  # initialize synthetic images from random noise or real images

# Differentiable Siamese Augmentation
CFG.dsa = True  # whether to use differentiable Siamese augmentation
CFG.dsa_strategy = 'color_crop_cutout_flip_scale_rotate'  # differentiable Siamese augmentation strategy

# Paths
CFG.data_path = '../dataset/'  # dataset path
CFG.buffer_path = '../buffer_storage/'  # buffer path

# Expert epochs and synthetic data steps
CFG.expert_epochs = 2  # how many expert epochs the target params are
CFG.syn_steps = 80  # how many steps to take on synthetic data

# Start epochs
CFG.max_start_epoch = 25  # max epoch we can start at
CFG.min_start_epoch = 0  # min epoch we can start at

# ZCA whitening
CFG.zca = True  # do ZCA whitening (use True if action='store_true')

# Load all expert trajectories into RAM
CFG.load_all = False  # only use if you can fit all expert trajectories into RAM (use True if action='store_true')

# Turn off differential augmentation during distillation
CFG.no_aug = False  # this turns off diff aug during distillation

# Distill textures instead
CFG.texture = False  # will distill textures instead (use True if action='store_true')
CFG.canvas_size = 2  # size of synthetic canvas
CFG.canvas_samples = 1  # number of canvas samples per iteration

# Number of expert files to read (leave as None unless doing ablations)
CFG.max_files = None

# Number of experts to read per file (leave as None unless doing ablations)
CFG.max_experts = None

# Force saving images for 50ipc
CFG.force_save = False  # this will save images for 50ipc (use True if action='store_true')
CFG.ema_decay = 0.999


# Learning rate for 'y' 
CFG.lr_y = 2.
# Momentum for 'y'
CFG.Momentum_y = 0.9

# WanDB Project Name
CFG.project = 'TEST'

# Threshold
CFG.threshold = 1.0

# Record loss
CFG.record_loss = False

# Sequential Generation
CFG.Sequential_Generation = True
CFG.expansion_end_epoch = 3000
CFG.current_max_start_epoch = 20


# Skip first evaluation
CFG.skip_first_eva = False  # If skip first eva 

# Parallel evaluation
CFG.parall_eva = False  # If parallel eva 

CFG.lr_lr = 0.00001

CFG.res = 32

CFG.device = [0]

CFG.Initialize_Label_With_Another_Model = False
CFG.Initialize_Label_Model = ""
CFG.Initialize_Label_Model_Dir = ""
CFG.Label_Model_Timestamp = -1

# long tail
CFG.imbalance_rate = 0.005
CFG.noise = False
CFG.duplicate = False
CFG.inherent = False
CFG.hard_label = False
CFG.longtailipc = False

# match stage
CFG.expert_dir = None
CFG.first_stage_expert_dir = None
CFG.second_stage_expert_dir = None

CFG.match_first_stage = False # whether to match first stage
CFG.first_stage_expert = False # almost always False, only for args check
CFG.match_classifier = True # exclusive to match_first_stage
CFG.partial_classifier = False # for both first and second stage
CFG.match_backbone = False # exclusive to match_first_stage
CFG.cls_t = None

CFG.match_second_stage = False # whether to match second stage

# 2nd stage initailization
CFG.load_path = None
CFG.load_it = None
CFG.Second_stage_Iteration = 5000
CFG.syn_steps_second = 80

# only used for baseline comparison
CFG.align_with_long_tail = False

# weighted loss used only for student trajectory
CFG.weighted_loss = False
CFG.second_stage_weighted_loss = False
CFG.first_weight_factor = 0.
CFG.second_weight_factor = 0.

CFG.BFLoss = False

# second stage epoch
CFG.second_min_start_epoch = 0
CFG.second_current_max_start_epoch = 0
CFG.second_max_start_epoch = 1
# CFG.match_second_classifier = True

CFG.first_stage_lambda = 0.5
CFG.second_stage_lambda = 0.5

CFG.syn_steps_second = 80
# CFG.second_lr_img = 1000
# CFG.second_lr_lr = 0.00001
# CFG.second_lr_y = 2.
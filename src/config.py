import os

# Training Hyperparameters
NUM_CLASSES         = 200
BATCH_SIZE          = 512
VAL_EVERY_N_EPOCH   = 1

NUM_EPOCHS          = 50
# OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.005, 'momentum': 0.9}
OPTIMIZER_PARAMS    = {'type': 'SGD', 'lr': 0.1, 'momentum': 0.9}
# OPTIMIZER_PARAMS    = {'type': 'AdamW', 'lr': 1e-2, 'betas': (0.9, 0.999), 'weight_decay': 0.01}
# SCHEDULER_PARAMS    = {'type': 'MultiStepLR', 'milestones': [30, 35], 'gamma': 0.2}
SCHEDULER_PARAMS    = {'type': 'ReduceLROnPlateau', 'monitor': 'loss/val', 'mode': 'min', 'factor': 0.2,
                       'patience': 10, 'min_lr': 1e-6, 'verbose': True}

# Dataaset
DATASET_ROOT_PATH   = 'datasets/'
NUM_WORKERS         = 8

# Augmentation
IMAGE_ROTATION      = 20
IMAGE_FLIP_PROB     = 0.5
IMAGE_NUM_CROPS     = 64
IMAGE_PAD_CROPS     = 4
IMAGE_MEAN          = [0.4802, 0.4481, 0.3975]
IMAGE_STD           = [0.2302, 0.2265, 0.2262]

NUM_GPUS            = 3

# Network
# MODEL_NAME          = 'MyNetwork'
# MODEL_NAME          = 'alexnet'
# MODEL_NAME          = 'resnet18'
# MODEL_NAME          = 'resnet34'
# MODEL_NAME          = 'resnet50'
# MODEL_NAME          = 'resnet101'
# MODEL_NAME          = 'resnet152'
# MODEL_NAME          = 'convnext_base'
# MODEL_NAME          = 'convnext_large'
# MODEL_NAME          = 'convnext_small'
MODEL_NAME          = 'convnext_tiny'
# MODEL_NAME          = 'deeplabv3_mobilenet_v3_large'
# MODEL_NAME          = 'deeplabv3_resnet101'
# MODEL_NAME          = 'deeplabv3_resnet50'
# MODEL_NAME          = 'densenet121'
# MODEL_NAME          = 'densenet161'
# MODEL_NAME          = 'densenet169'
# MODEL_NAME          = 'densenet201'
# MODEL_NAME          = 'efficientnet_b0'
# MODEL_NAME          = 'efficientnet_b1'
# MODEL_NAME          = 'efficientnet_b2'
# MODEL_NAME          = 'efficientnet_b3'
# MODEL_NAME          = 'efficientnet_b4'
# MODEL_NAME          = 'efficientnet_b5'
# MODEL_NAME          = 'efficientnet_b6'
# MODEL_NAME          = 'efficientnet_b7'
# MODEL_NAME          = 'efficientnet_v2_l'
# MODEL_NAME          = 'efficientnet_v2_m'
# MODEL_NAME          = 'efficientnet_v2_s'
# MODEL_NAME          = 'fasterrcnn_mobilenet_v3_large_320_fpn'
# MODEL_NAME          = 'fasterrcnn_mobilenet_v3_large_fpn'
# MODEL_NAME          = 'fasterrcnn_resnet50_fpn'
# MODEL_NAME          = 'fasterrcnn_resnet50_fpn_v2'
# MODEL_NAME          = 'fcn_resnet101'
# MODEL_NAME          = 'fcn_resnet50'
# MODEL_NAME          = 'fcos_resnet50_fpn'
# MODEL_NAME          = 'googlenet'
# MODEL_NAME          = 'inception_v3'
# MODEL_NAME          = 'keypointrcnn_resnet50_fpn'
# MODEL_NAME          = 'lraspp_mobilenet_v3_large'
# MODEL_NAME          = 'maskrcnn_resnet50_fpn'
# MODEL_NAME          = 'maskrcnn_resnet50_fpn_v2'
# MODEL_NAME          = 'maxvit_t'
# MODEL_NAME          = 'mc3_18'
# MODEL_NAME          = 'mnasnet0_5'
# MODEL_NAME          = 'mnasnet0_75'
# MODEL_NAME          = 'mnasnet1_0'
# MODEL_NAME          = 'mnasnet1_3'
# MODEL_NAME          = 'mobilenet_v2'
# MODEL_NAME          = 'mobilenet_v3_large'
# MODEL_NAME          = 'mobilenet_v3_small'
# MODEL_NAME          = 'mvit_v1_b'
# MODEL_NAME          = 'mvit_v2_s'
# MODEL_NAME          = 'r2plus1d_18'
# MODEL_NAME          = 'r3d_18'
# MODEL_NAME          = 'raft_large'
# MODEL_NAME          = 'raft_small'
# MODEL_NAME          = 'regnet_x_16gf'
# MODEL_NAME          = 'regnet_x_1_6gf'
# MODEL_NAME          = 'regnet_x_32gf'
# MODEL_NAME          = 'regnet_x_3_2gf'
# MODEL_NAME          = 'regnet_x_400mf'
# MODEL_NAME          = 'regnet_x_800mf'
# MODEL_NAME          = 'regnet_x_8gf'
# MODEL_NAME          = 'regnet_y_128gf'
# MODEL_NAME          = 'regnet_y_16gf'
# MODEL_NAME          = 'regnet_y_1_6gf'
# MODEL_NAME          = 'regnet_y_32gf'
# MODEL_NAME          = 'regnet_y_3_2gf'
# MODEL_NAME          = 'regnet_y_400mf'
# MODEL_NAME          = 'regnet_y_800mf'
# MODEL_NAME          = 'regnet_y_8gf'
# MODEL_NAME          = 'resnext50_32x4d'
# MODEL_NAME          = 'resnext101_32x8d'
# MODEL_NAME          = 'resnext101_64x4d'
# MODEL_NAME          = 'retinanet_resnet50_fpn'
# MODEL_NAME          = 'retinanet_resnet50_fpn_v2'
# MODEL_NAME          = 's3d'
# MODEL_NAME          = 'shufflenet_v2_x0_5'
# MODEL_NAME          = 'shufflenet_v2_x1_0'
# MODEL_NAME          = 'shufflenet_v2_x1_5'
# MODEL_NAME          = 'shufflenet_v2_x2_0'
# MODEL_NAME          = 'squeezenet1_0'
# MODEL_NAME          = 'squeezenet1_1'
# MODEL_NAME          = 'ssd300_vgg16'
# MODEL_NAME          = 'ssdlite320_mobilenet_v3_large'
# MODEL_NAME          = 'swin3d_b'
# MODEL_NAME          = 'swin3d_s'
# MODEL_NAME          = 'swin3d_t'
# MODEL_NAME          = 'swin_b'
# MODEL_NAME          = 'swin_s'
# MODEL_NAME          = 'swin_t'
# MODEL_NAME          = 'swin_v2_b'
# MODEL_NAME          = 'swin_v2_s'
# MODEL_NAME          = 'swin_v2_t'
# MODEL_NAME          = 'vgg11'
# MODEL_NAME          = 'vgg11_bn'
# MODEL_NAME          = 'vgg13'
# MODEL_NAME          = 'vgg13_bn'
# MODEL_NAME          = 'vgg16'
# MODEL_NAME          = 'vgg16_bn'
# MODEL_NAME          = 'vgg19'
# MODEL_NAME          = 'vgg19_bn'
# MODEL_NAME          = 'vit_b_16'
# MODEL_NAME          = 'vit_b_32'
# MODEL_NAME          = 'vit_h_14'
# MODEL_NAME          = 'vit_l_16'
# MODEL_NAME          = 'vit_l_32'
# MODEL_NAME          = 'wide_resnet50_2'
# MODEL_NAME          = 'wide_resnet101_2'

# Compute related
ACCELERATOR         = 'gpu'
DEVICES             = [3,4,5]
PRECISION_STR       = '32-true'

# Logging
WANDB_PROJECT       = 'aue8088-pa1'
WANDB_ENTITY        = os.environ.get('WANDB_ENTITY')
WANDB_SAVE_DIR      = 'wandb/'
WANDB_IMG_LOG_FREQ  = 50
WANDB_NAME          = f'{MODEL_NAME}-B{BATCH_SIZE}-{OPTIMIZER_PARAMS["type"]}'
WANDB_NAME         += f'-{SCHEDULER_PARAMS["type"]}{OPTIMIZER_PARAMS["lr"]:.1E}'

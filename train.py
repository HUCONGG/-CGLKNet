import argparse
import os
from collections import OrderedDict
from glob import glob
import random
import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations
from albumentations import Compose
from albumentations.augmentations import transforms
from albumentations.augmentations import geometric

from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
from albumentations import RandomRotate90, Resize
from torch.cuda.amp import autocast, GradScaler
import archs
import losses
from dataset import Dataset
import sys

from utils import AverageMeter, str2bool


from ACDC_dataset import ACDCDataset
from dataset_ACDC import ACDC_dataset,RandomGenerator

from tensorboardX import SummaryWriter
import shutil
import subprocess
import os
ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')
from oct import oct_resnet50_fusion
def list_type(s):
    str_list = s.split(',')
    int_list = [int(a) for a in str_list]
    return int_list
from datetime import datetime

def parse_args(lr,b,data_name,model_name,keyword = ''):
    start = datetime.now().strftime("%m-%d-%H-%M")
    model_names = model_name + keyword
    data_name = data_name
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=f'{start}_{lr}_{b}',
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default = 200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=b, type=int,
                        metavar='N', help='mini-batch size (default: 16)')

    parser.add_argument('--dataseed', default=2907, type=int,
                        help='')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default=model_names)
    
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=224, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=224, type=int,
                        help='image height')
    parser.add_argument('--input_list', type=list_type, default=[128, 160, 256])

    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default=data_name, help='dataset name')      
    parser.add_argument('--data_dir', default='/home/hucon/vscode/u-kan/Seg_UKAN/inputs', help='dataset dir')

    parser.add_argument('--output_dir', default=f'/home/hucon/vscode/u-kan/Seg_UKAN/new_output/{data_name}/{model_names}', help='ouput dir')


    # optimizer
    parser.add_argument('--optimizer', default='Adam',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')

    parser.add_argument('--lr', '--learning_rate', default=lr, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=0, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')

    parser.add_argument('--kan_lr', default=1e-2, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--kan_weight_decay', default=1e-4, type=float,
                        help='weight decay')

    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    parser.add_argument('--cfg', type=str, metavar="FILE", help='path to config file', )

    parser.add_argument('--num_workers', default=0, type=int)

    config = parser.parse_args()

    return config

# weights = nn.Parameter(torch.ones(4))
# # 计算权重的softmax值

# weights_softmax = F.softmax(weights, dim=0)
def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}

    model.train()
        # 定义一个包含四个元素的可学习参数向量

    pbar = tqdm(total=len(train_loader))
    for input, target, _ in train_loader:
        # print(input.shape, target.shape)
        input = input.cuda()
        target = target.cuda()
        loss = 0
        # compute output
        if config['deep_supervision']:
             
            P = model(input, mode='train')
            # print(len(P))
            if  not isinstance(P, list):
                P = [P]
            n_outs = len(P)
            out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
            if config['deep_supervision'] == 'mutation':
                ss = [x for x in powerset(out_idxs)]
            elif config['deep_supervision'] == True:
                ss = [[x] for x in out_idxs]
            else:
                ss = [[-1]]
            # print(ss)
            
            loss = 0.0
            w_ce, w_dice = 0.3, 0.7
          
            for s in ss:
                iout = 0.0
                if(s==[]):
                    continue
                for idx in range(len(s)):
                    iout += P[s[idx]]
                from torch.nn.modules.loss import CrossEntropyLoss
                ce_loss = CrossEntropyLoss()
                
                sys.path.append('/home/hucon/vscode/u-kan/Seg_UKAN/make_models/EMCAD-main/utils')
                from utils_copy import powerset, one_hot_encoder, DiceLoss, val_single_volume
                dice_loss = DiceLoss(1)
                loss_ce = ce_loss(iout, target)
                # target = target.squeeze(2)
                # print('target',target.shape)
                loss_dice = dice_loss(iout, target, softmax=True)
                loss += (w_ce * loss_ce + w_dice * loss_dice)
                output = iout
            
            else:
                outputs = model(input)
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
            # iou, dice, _ = iou_score(output, target)
            acc,iou, dice, hd, hd95, recall, specificity, precision = indicators(output, target)
        else:
            outputs = model(input)

            loss = criterion(outputs, target)
            iou, dice, _ = iou_score(outputs, target)
            # acc, iou, dice, hd_, hd95_, recall_, specificity_, precision_ = indicators_multiclass(outputs, target)
        # compute gradient and do optimizing step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        # 假设你的优化器是 optimizer
        current_lr = optimizer.param_groups[0]['lr']

        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
            ('lr', current_lr), 
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'accuracy': AverageMeter(),
                  'iou': AverageMeter(),
                   'dice': AverageMeter(),
                   'hd': AverageMeter(),
                    'hd95': AverageMeter(),
                    'recall': AverageMeter(),
                    'specificity': AverageMeter(),
                    'precision': AverageMeter()
    
                   }

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, _ in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
            
                # if model_name ==  'EMCADN':
                    
                P = model(input, mode='train')
                # print(len(P))
                if  not isinstance(P, list):
                    P = [P]
                n_outs = len(P)
                out_idxs = list(np.arange(n_outs)) #[0, 1, 2, 3]#, 4, 5, 6, 7]
                if config['deep_supervision'] == 'mutation':
                    ss = [x for x in powerset(out_idxs)]
                elif config['deep_supervision'] == True:
                    ss = [[x] for x in out_idxs]
                else:
                    ss = [[-1]]
                # print(ss)
                
                loss = 0.0
                w_ce, w_dice = 0.3, 0.7
            
                for s in ss:
                    iout = 0.0
                    if(s==[]):
                        continue
                    for idx in range(len(s)):
                        iout += P[s[idx]]
                    from torch.nn.modules.loss import CrossEntropyLoss
                    ce_loss = CrossEntropyLoss()
                    
                    sys.path.append('/home/hucon/vscode/u-kan/Seg_UKAN/make_models/EMCAD-main/utils')
                    from utils_copy import powerset, one_hot_encoder, DiceLoss, val_single_volume
                    dice_loss = DiceLoss(1)
                    loss_ce = ce_loss(iout, target)
                    # target = target.squeeze(2)
                    # print('target',target.shape)
                    loss_dice = dice_loss(iout, target, softmax=True)
                    loss += (w_ce * loss_ce + w_dice * loss_dice)
                    output = iout
                    # 计算加权损失
                else:
                    outputs = model(input)
                    for output in outputs:
                        loss += criterion(output, target)
                    loss /= len(outputs)
                # iou, dice, hd, hd95, recall, specificity, precision = iou_score(output, target)
                acc, iou, dice, hd, hd95, recall, specificity, precision = indicators(output, target)
            else:
                outputs = model(input)
                loss = criterion(outputs, target)
                acc,iou, dice, hd, hd95, recall, specificity, precision = indicators(outputs, target)
                # acc,iou, dice, hd, hd95, recall, specificity, precision = indicators_multiclass(outputs, target)
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['accuracy'].update(acc, input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice'].update(dice, input.size(0))
            avg_meters['hd'].update(hd, input.size(0))
            avg_meters['hd95'].update(hd95, input.size(0))
            avg_meters['recall'].update(recall, input.size(0))
            avg_meters['specificity'].update(specificity, input.size(0))
            avg_meters['precision'].update(precision, input.size(0))


            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
                ('dice', avg_meters['dice'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()


    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('accuracy', avg_meters['accuracy'].avg),
                        ('iou', avg_meters['iou'].avg),
                        ('dice', avg_meters['dice'].avg),
                        ('hd', avg_meters['hd'].avg),
                        ('hd95', avg_meters['hd95'].avg),
                        ('recall', avg_meters['recall'].avg),
                        ('specificity', avg_meters['specificity'].avg),
                        ('precision', avg_meters['precision'].avg)
                        
                        ])

def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

###############
def save_checkpoint(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)
###############
def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    new_path = remove_last_two_parts(filename)
    return model, optimizer, epoch, new_path
###############
import segmentation_models_pytorch as smp

def create_model(model_name):
    if model_name == 'Msa2Net_mx_y':
        return Msa2Net_mx_y(num_classes=1)

###############
def main(lr,b,data_name,model_name,keyword = '',epoch = 200,size = 224):
    seed_torch()
    config = vars(parse_args(lr,b,data_name,model_name,keyword))
    config['input_w']  = size
    config['input_h'] =  size
    exp_name = config.get('name')
    output_dir = config.get('output_dir')
    # dataset_name = config.get('dataset')
    my_writer = SummaryWriter(f'{output_dir}/{exp_name}')
    config['epochs'] = epoch
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    with open(f'{output_dir}/{exp_name}/config.yml', 'w') as f:
        yaml.dump(config, f)
    from sklearn.model_selection import KFold
    dataset_name = config['dataset']
    img_ext = '.png'
    mask_ext = '.png'
    if dataset_name == 'busi':
        mask_ext = '_mask.png'
    elif dataset_name == 'glas':
        mask_ext = '.png'
    elif dataset_name == 'cvc':
        mask_ext = '.png'
    elif dataset_name == 'isic_2018':
        mask_ext = '.jpg'
        img_ext = '.jpg'
    elif dataset_name == 'MoNuSeg':
        mask_ext = '.png'
    elif data_name == 'DRIVE':
        mask_ext = '.tif'
        img_ext = '.tif'
    elif dataset_name == 'NEW_BUSI':
        mask_ext = '_mask.png'
    elif dataset_name == 'isic_2016':
        mask_ext = '_Segmentation.png'
        img_ext = '.jpg'
    elif dataset_name == 'peng':
        img_ext = '.tif'
        mask_ext = ".png"
    elif dataset_name == 'CAMUS2D':
        img_ext = '.jpg'
        mask_ext = '.jpg'
    if config['loss'] == 'BCEWithLogitsLoss':
            criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()
        
        cudnn.benchmark = True 
        if  model_name == 'UKAN_MSCA_KAN':
            model =UKAN_MSCA_S_C_KAN(num_classes=1, input_channels = 3)
        if model_name == 'UKAN':
            model = UKAN(num_classes=1, input_channels = 3)
        if  model_name == 'UKAN_MSCA_S_C_KAN_GATE_DS':
            model =UKAN_MSCA_S_C_KAN_GATE_DS(num_classes=1, input_channels = 3)
        if model_name == 'DuAT':
            model = DuAT().to('cuda')
        if model_name == 'MSCA_pvt_S_C_KAN_GATE_DS':
            model = MSCA_pvt_S_C_KAN_GATE_DS(num_classes=1, input_channels = 3,)
        # if model_name == 'EMCADNet':
        #     model = EMCADNet().cuda()
        if model_name == 'Msa2Net':
            model = Msa2Net(num_classes=1)
        if model_name == 'Msa2Net_mx_y':
            model = Msa2Net_mx_y(num_classes=1)
        if model_name == 'UKAN_MSCA_S_C_KAN_MGATE_DS':
            model = UKAN_MSCA_S_C_KAN_MGATE_DS(num_classes=1, input_channels = 3)
        if model_name == 'UMA_G_U_K':
            model = UMA_G_U_K(num_classes=1, input_channels = 3)
        if model_name == 'mdsknet':
            model = mdsknet(num_classes=1, input_channels = 3)
        if model_name == 'umdslnet':
            model = umdslnet(num_classes=1, input_channels = 3)
        if model_name == 'UKAN_MSCA_dls_KAN_GATE_DS':
            model = UKAN_MSCA_dls_KAN_GATE_DS(num_classes=1, input_channels = 3)
        import segmentation_models_pytorch as smp
        if model_name == 'unet++':
            model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=
            1,
        )  
        if model_name == 'unet':
            model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights=None,
            in_channels=3,
            classes=1,
        )
        if model_name == 'pvt_deconvNet':
            model = pvt_deconvNet(num_classes=1)
        if model_name == 'mdunet':
            model = mdunet(
            encoder_name="resnet34",
            encoder_weights='imagenet',
            in_channels=3,
            classes=1,
        ) 
        if model_name == 'Msa2Net_LKA':
            model = Msa2Net_LKA(num_classes=1)
        if model_name == 'Msa2Net_2_2_DL_L':
            model = Msa2Net_2_2_DL_L(num_classes=1)
        if model_name == 'Msa2Net_3_1_DL_L':
            model = Msa2Net_3_1_DL_L(num_classes=1)
        if model_name =='Rolling_L':
            model = Rolling_Unet_L(num_classes=1)
        if model_name == 'M2SNet':
            model = M2SNet()
        if model_name =='EMCADNet':
            model = EMCADNet()
        if model_name == 'TransAttUnet':
            model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=1)
        from albumentations import Compose, RandomRotate90, Flip, Resize, Normalize, RandomBrightnessContrast, GaussNoise, ElasticTransform, RandomCrop
        from albumentations import ShiftScaleRotate
        train_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            RandomRotate90(),
            # geometric.transforms.Flip(),
            albumentations.Flip(),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=0),
            transforms.Normalize(),
        ], is_check_shapes=False)

        val_transform = Compose([
            Resize(config['input_h'], config['input_w']),
            transforms.Normalize(),
        ], is_check_shapes=False)
    if config['dataset'] == 'cvc':
        singword ='train_orin'
    else:
        singword ='train'
    img_ids = sorted(glob(os.path.join(config['data_dir'], config['dataset'],singword,'images', '*' + img_ext)))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    kf = KFold(n_splits=5, random_state=config['dataseed'], shuffle=True)
    if singal == 0:
        for fold, (train_index, val_index) in enumerate(kf.split(img_ids)):

            if fold not in [1,2,3,4]:
                continue
            if config['loss'] == 'BCEWithLogitsLoss':
                criterion = nn.BCEWithLogitsLoss().cuda()
            else:
                criterion = losses.__dict__[config['loss']]().cuda()
            
            cudnn.benchmark = True 
            if  model_name == 'UKAN_MSCA_KAN':
                model =UKAN_MSCA_S_C_KAN(num_classes=1, input_channels = 3)
            if model_name == 'UKAN':
                model = UKAN(num_classes=1, input_channels = 3)
            if  model_name == 'UKAN_MSCA_S_C_KAN_GATE_DS':
                model =UKAN_MSCA_S_C_KAN_GATE_DS(num_classes=1, input_channels = 3)
            if model_name == 'DuAT':
                model = DuAT().to('cuda')
            if model_name == 'MSCA_pvt_S_C_KAN_GATE_DS':
                model = MSCA_pvt_S_C_KAN_GATE_DS(num_classes=1, input_channels = 3,)
            # if model_name == 'EMCADNet':
            #     model = EMCADNet().cuda()
            if model_name == 'Msa2Net':
                model = Msa2Net(num_classes=1)
            if model_name == 'Msa2Net_mx_y':
                model = Msa2Net_mx_y(num_classes=1)
            if model_name == 'UKAN_MSCA_S_C_KAN_MGATE_DS':
                model = UKAN_MSCA_S_C_KAN_MGATE_DS(num_classes=1, input_channels = 3)
            if model_name == 'UMA_G_U_K':
                model = UMA_G_U_K(num_classes=1, input_channels = 3)
            if model_name == 'mdsknet':
                model = mdsknet(num_classes=1, input_channels = 3)
            if model_name == 'umdslnet':
                model = umdslnet(num_classes=1, input_channels = 3)
            if model_name == 'UKAN_MSCA_dls_KAN_GATE_DS':
                model = UKAN_MSCA_dls_KAN_GATE_DS(num_classes=1, input_channels = 3)
            import segmentation_models_pytorch as smp
            if model_name == 'unet++':
                model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights='imagenet',
                in_channels=3,
                classes=
                1,
            )  
            if model_name == 'unet':
                model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            if model_name == 'pvt_deconvNet':
                model = pvt_deconvNet(num_classes=1)
            if model_name == 'mdunet':
                model = mdunet(
                encoder_name="resnet34",
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
            ) 
            if model_name == 'Msa2Net_LKA':
                model = Msa2Net_LKA(num_classes=1)
            if model_name == 'UNext':
                model = UNext(1)
            if model_name =='ACC_UNet':
                model = ACC_UNet(3,1)
            if model_name == 'MaxViT_deformableLKAFormer':
                model = MaxViT_deformableLKAFormer(num_classes=1)
            if model_name =='Rolling_L':
                model = Rolling_Unet_L(num_classes=1)
            if model_name == 'msa2net_nc_nd':
                model = msa2net_nc_nd(num_classes=1)
            if  model_name == 'msa2net_nc_yd':
                model = msa2net_nc_yd(num_classes=1)
            if model_name == 'msa2net_nm_nd':
                model = msa2net_nm_nd(num_classes=1)
            if model_name == 'msa2net_nm_yd':
                model = msa2net_nm_yd(num_classes=1)
            if  model_name == 'msa2net_yc_nd':
                model =  msa2net_yc_nd(num_classes=1)
            if  model_name == 'msa2net_ym_nd':
                model =  msa2net_ym_nd(num_classes=1)
#################
            if model_name == 'msa2net3x3':
                model = msa2net_3x3(num_classes=1)
                
            if model_name == 'msa2net5x3':
                model = msa2net_5x3(num_classes=1)
                
            if model_name == 'msa2net5x5':
                model = msa2net_3x3(num_classes=1)
                
            if model_name == 'msa2net7x7':
                model = msa2net_7x7(num_classes=1)
                
            if model_name == 'msa2net9x7':
                model = msa2net_9x7(num_classes=1)
                
            if model_name == 'msa2net9x9':
                model = msa2net_9x9(num_classes=1)
##############
            if model_name == 'M2SNet':
                model = M2SNet()
            if model_name =='EMCADNet':
                model = EMCADNet()
            if model_name == 'oct':
                model = oct_resnet50_fusion(pretrained=True)
            if model_name == 'TransAttUnet':
                model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=1)
            model = model.cuda()
            print(f"Processing fold {fold}")
            os.makedirs(f'{output_dir}/{exp_name}/fold_{fold}', exist_ok=True)
            path = f'{output_dir}/{exp_name}/fold_{fold}'
            train_img_ids = [img_ids[i] for i in train_index]
            val_img_ids = [img_ids[i] for i in val_index]
            txt_path = f'{output_dir}/{exp_name}/fold_{fold}'
            with open(f'{txt_path}/val_img_ids.txt', 'w') as f:
                for img_id in val_img_ids:
                    f.write(f'{img_id}\n')
            

            # img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
            # path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset_name}/train_orin/images/*{img_ext}'
            # print('path',path)
            # train_img_ids, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=config['dataseed'])

            print('train_img_ids',len(train_img_ids))
            print('val_img_ids',len(val_img_ids))

            import thop
            # input = torch.rand(1, 3, 224, 224).cuda()
            # flops, params = thop.profile(model, inputs=(input, ))
            # print(f"Flops: {flops/1e9} G, Params: {params/1e6} M")
            param_groups = []
            for name, param in model.named_parameters():
                if 'kan' in name.lower() and 'fc' in name.lower():
                    param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
                else:
                    param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  

            if config['optimizer'] == 'Adam':
                optimizer = optim.Adam(param_groups)


            elif config['optimizer'] == 'SGD':
                optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
            else:
                raise NotImplementedError

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
            elif config['scheduler'] == 'MultiStepLR':
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
            elif config['scheduler'] == 'ConstantLR':
                scheduler = None
            else:
                raise NotImplementedError

            shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('deform_lsk.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('/home/hucon/vscode/u-kan/Seg_UKAN/msa2net_networks/msa2net_mx_y.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('/home/hucon/vscode/u-kan/Seg_UKAN/denet.py', f'{output_dir}/{exp_name}/')

            # Data loading code
            if dataset_name == 'MoNuSeg':
                from albumentations import Compose, RandomRotate90, Flip, Resize, Normalize, RandomBrightnessContrast, GaussNoise, ElasticTransform, RandomCrop
                from albumentations import ShiftScaleRotate
                train_transform = Compose([
                    Resize(config['input_h'], config['input_w']),
                    RandomRotate90(),
                    # geometric.transforms.Flip(),
                    albumentations.Flip(),

                    ShiftScaleRotate(shift_limit=0.5, scale_limit=0.5, rotate_limit=0), 
                    transforms.Normalize(),
                ])
            else:
                from albumentations import Compose, RandomRotate90, Flip, Resize, Normalize, RandomBrightnessContrast, GaussNoise, ElasticTransform, RandomCrop
                from albumentations import ShiftScaleRotate


            train_dataset = Dataset(
                img_ids=train_img_ids,
                img_dir=os.path.join(config['data_dir'], config['dataset'],singword, 'images'),
                mask_dir=os.path.join(config['data_dir'], config['dataset'], singword,'masks'),
                img_ext=img_ext,
                mask_ext=mask_ext,
                num_classes=config['num_classes'],
                transform=train_transform)

            val_dataset = Dataset(
                img_ids=val_img_ids,
                img_dir=os.path.join(config['data_dir'] ,config['dataset'], singword,'images'),
                mask_dir=os.path.join(config['data_dir'], config['dataset'],singword, 'masks'),
                img_ext=img_ext,
                mask_ext=mask_ext,
                num_classes=config['num_classes'],
                transform=val_transform)
            
            
            train_loader = torch.utils.data.DataLoader(
                        train_dataset,
                        batch_size=config['batch_size'],
                        shuffle=True,
                        num_workers=config['num_workers'],
                        drop_last=False)
            val_loader = torch.utils.data.DataLoader(
                        val_dataset,
                        batch_size=config['batch_size'],
                        shuffle=False,
                        num_workers=config['num_workers'],
                        drop_last=False)
            #######################
            log = OrderedDict([
            ('epoch', []),
            ('lr', []),
            ('loss', []),
            ('iou', []),
            ('val_loss', []),
            ('val_acc', []),
            ('val_iou', []),
            ('val_dice', []),
            ('best_val_iou',[]),
            ('hd95', []),
            ('recall', []),
            ('specificity', []),
            ('precision', [])
                ])
            start_epoch = 0
            if continue_training == 1:
                model, optimizer, epoch,new_path = load_checkpoint(continue_path, model, optimizer)
                start_epoch  = epoch+1
                exp_name = new_path
                print(f"Loaded model from epoch {start_epoch}") 

            best_iou = 0
            best_dice= 0
            trigger = 0
            os.makedirs(f'{output_dir}/{exp_name}/epoch', exist_ok=True)
            os.makedirs(f'{output_dir}/{exp_name}', exist_ok=True)
            best_model = None
            for epoch in range(start_epoch,config['epochs']):
                # epoch = epoch + start_epoch
                print('Epoch [%d/%d]' % (epoch, config['epochs']))

                # train for one epoch
                train_log = train(config, train_loader, model, criterion, optimizer)
                # evaluate on validation set
                val_log = validate(config, val_loader, model, criterion)

                if config['scheduler'] == 'CosineAnnealingLR':
                    scheduler.step()
                elif config['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(val_log['loss'])

                print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                    % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

                log['epoch'].append(epoch)
                log['lr'].append(round(config['lr'], 5))
                log['loss'].append(round(train_log['loss'], 5))
                log['iou'].append(round(train_log['iou'], 5))
                log['val_loss'].append(round(val_log['loss'], 5))
                log['val_acc'].append(round(val_log['accuracy'], 5))
                log['val_iou'].append(round(val_log['iou'], 5))
                log['val_dice'].append(round(val_log['dice'], 5))
                log['best_val_iou'].append(round(best_iou, 5))
                log['hd95'].append(round(val_log['hd95'], 5))
                log['recall'].append(round(val_log['recall'], 5))
                log['specificity'].append(round(val_log['specificity'], 5))
                log['precision'].append(round(val_log['precision'], 5))
                if continue_training == 1:
                    pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/continue_log.csv', index=False)
                else:
                    pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/fold_{fold}/log.csv', index=False)

                my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
                my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
                my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
                my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
                my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

                my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
                my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)
                trigger += 1
                os.makedirs(f'{output_dir}/{exp_name}/fold_{fold}/epoch', exist_ok=True)
                ###create a txt file to save the val_img_ids
                # df  = pd.DataFrame(val_img_ids)
                # with open('val_img_ids.txt', 'w') as f:
                #     for img_id in val_img_ids:
                #         f.write(f'{img_id}\n')
                if (epoch+1) in [299,epoch]:
                    iou = val_log['iou']
                    if continue_training == 1:
                        save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                    else:
                        save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/fold_{fold}/epoch/{epoch}_{iou}model.pth')
                
                if val_log['iou'] > best_iou:
                    best_iou_epoch = 0
                    best_iou = val_log['iou']
                    # torch.save(model.state_dict(), f'{output_dir}/{exp_name}/{epoch}_{best_iou}model.pth')
                    
                    best_dice = val_log['dice']
                    best_iou_epoch = epoch
                    best_model = model.state_dict()
                    # if config['dataset'] =='NEW_BUSI' and best_iou > 0.66 and epoch > 200:
                    #     torch.save(model.state_dict(), f'{path}/{best_iou_epoch}_{best_iou}model.pth')
                    if config['dataset'] == 'NEW_BUSI' and best_iou > 0.74:
                        torch.save(model.state_dict(), f'{path}/{best_iou_epoch}_{best_iou}model.pth')
                    if config['dataset'] == 'glas'and best_iou > 0.86:
                        torch.save(model.state_dict(), f'{path}/{best_iou_epoch}_{best_iou}model.pth')
                    # if config['dataset'] == 'cvc'and best_iou > 0.87:
                    #     torch.save(model.state_dict(), f'{path}/{best_iou_epoch}_{best_iou}model.pth')
                    print(f"best model:IOU:{best_iou}, DICE:{best_dice},{best_iou_epoch}")
                    # trigger = 0
                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break
            best_path = f'{output_dir}/{exp_name}/fold_{fold}/{best_iou_epoch}_{best_iou}best_model.pth'
            torch.save(best_model, f'{output_dir}/{exp_name}/fold_{fold}/{best_iou_epoch}_{best_iou}best_model.pth')
            bathch_size = [10]
            for i, b in enumerate(bathch_size):
                if data_name in ['isic_2016','GLAS_orin']:
                    key_f = 0
                else:
                    key_f = 1
                a, b, c, d, e, f, g,m_path = test(
                    m_n = model_name,
                    data_path=data_name,
                    key_fold=key_f,
                    data_key=None,
                    batch_size=b,
                    m_path= best_path ,
                    val_img_ids=f'{txt_path}/val_img_ids.txt'
                )
                acc =[]
                iou = []
                dice = []
                recall = []
                specificity = []
                precision = []
                hd95 = []
                acc.append(a)
                iou.append(b)
                dice.append(c)
                recall.append(d)
                specificity.append(e)
                precision.append(f)
                hd95.append(g)

            # 计算均值和标准差
            metrics = {
                'acc': acc,
                'iou': iou,
                'dice': dice,
                'recall': recall,
                'specificity': specificity,
                'precision': precision,
                'hd95': hd95
            }
            our_output_dir = f'{output_dir}/{exp_name}/fold_{fold}/metrics.txt'
            os.makedirs(f'{output_dir}/{exp_name}/fold_{fold}', exist_ok=True)
            #save the metrics to a csv file
            with open(our_output_dir, 'w') as f:
                f.write("ACC\tIOU\tDSC\tRecall\tSp\tPrecision\tHD95↓\n")
                mean_std_values = []
                for key, values in metrics.items():
                    mean_value = np.mean(values)
                    std_value = np.std(values)
                    if key != 'hd95':
                        mean_value *= 100
                        std_value *= 100
                    mean_std_values.append(f'{mean_value:.2f} ± {std_value:.2f}')
                f.write("\t".join(mean_std_values) + '\n')

            # print("均值和标准差已保存到文件中{}".format(new_output_file))
            end = datetime.now().strftime("%Y-%m-%d %H:%M")
            # log.info('Best IoU: %.4f at epoch %d' % (best_iou, best_iou_epoch))
            print('end',end)
            print('Best IoU: %.4f at epoch %d' % (best_iou, best_iou_epoch))
                
    elif singal == 1:
            if  model_name == 'UKAN_MSCA_KAN':
                model =UKAN_MSCA_S_C_KAN(num_classes=1, input_channels = 3)
            if model_name == 'UKAN':
                model = UKAN(num_classes=1, input_channels = 3)
            if  model_name == 'UKAN_MSCA_S_C_KAN_GATE_DS':
                model =UKAN_MSCA_S_C_KAN_GATE_DS(num_classes=1, input_channels = 3)
            if model_name == 'DuAT':
                model = DuAT().to('cuda')
            if model_name == 'MSCA_pvt_S_C_KAN_GATE_DS':
                model = MSCA_pvt_S_C_KAN_GATE_DS(num_classes=1, input_channels = 3,)
            if model_name == 'EMCADNet':
                model = EMCADNet(num_classes=4).cuda()
            if model_name == 'Msa2Net':
                model = Msa2Net(num_classes=1)
            if model_name == 'Msa2Net_mx_y':
                model = Msa2Net_mx_y(num_classes=1)
            if model_name == 'UKAN_MSCA_S_C_KAN_MGATE_DS':
                model = UKAN_MSCA_S_C_KAN_MGATE_DS(num_classes=1, input_channels = 3)
            if model_name == 'UMA_G_U_K':
                model = UMA_G_U_K(num_classes=1, input_channels = 3)
            if model_name == 'mdsknet':
                model = mdsknet(num_classes=1, input_channels = 3)
            if model_name == 'DLK':
                model = MaxViT_deformableLKAFormer(num_classes=1)
            if model_name == 'umdslnet':
                model = umdslnet(num_classes=1, input_channels = 3)
            if model_name == 'UKAN_MSCA_dls_KAN_GATE_DS':
                model = UKAN_MSCA_dls_KAN_GATE_DS(num_classes=1, input_channels = 3)
            import segmentation_models_pytorch as smp
            if model_name == 'unet++':
                model = smp.UnetPlusPlus(
                encoder_name="resnet34",
                encoder_weights='imagenet',
                in_channels=3,
                classes=
                1,
            )  
            if model_name == 'unet':
                model = smp.Unet(
                encoder_name="resnet34",
                encoder_weights=None,
                in_channels=3,
                classes=1,
            )
            if model_name == 'pvt_deconvNet':
                model = pvt_deconvNet(num_classes=1)
            if model_name == 'mdunet':
                model = mdunet(
                encoder_name="resnet34",
                encoder_weights='imagenet',
                in_channels=3,
                classes=1,
            ) 
            if model_name == 'Msa2Net_LKA':
                model = Msa2Net_LKA(num_classes=1)
            if model_name == 'UNext':
                model = UNext(in_channels=3, num_classes=1)
            if model_name =='ACC_UNet':
                model = ACC_UNet(3,1)
            if model_name == 'MaxViT_deformableLKAFormer':
                model = MaxViT_deformableLKAFormer(num_classes=1)
            if model_name =='Rolling_L':
                model = Rolling_Unet_L(num_classes=1)
            if model_name == 'msa2net_nc_nd':
                model = msa2net_nc_nd(num_classes=1)
            if  model_name == 'msa2net_nc_yd':
                model = msa2net_nc_yd(num_classes=1)
            if  model_name == 'msa2net_yc_nd':
                model =  msa2net_yc_nd(num_classes=1)
            if  model_name == 'msa2net_ym_nd':
                model =  msa2net_ym_nd(num_classes=1)
            if  model_name == 'msa2net_nm_nd':
                model =  msa2net_nm_nd(num_classes=1)
            if model_name == 'M2SNet':
                model = M2SNet()
            if model_name =='EMCADNet':
                model = EMCADNet()
            if model_name == 'oct':
                model = oct_resnet50_fusion(pretrained=True)
            if model_name == 'TransAttUnet':
                model = UNet_Attention_Transformer_Multiscale(n_channels=3, n_classes=1)
            model = model.cuda()
            param_groups = []
            for name, param in model.named_parameters():
                if 'kan' in name.lower() and 'fc' in name.lower():
                    param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
                else:
                    param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  

            if config['optimizer'] == 'Adam':
                optimizer = optim.Adam(param_groups)


            elif config['optimizer'] == 'SGD':
                optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
            else:
                raise NotImplementedError

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
            elif config['scheduler'] == 'MultiStepLR':
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
            elif config['scheduler'] == 'ConstantLR':
                scheduler = None
            else:
                raise NotImplementedError
            dataset = config['dataset']
            train_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/train'
            val_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/validation'
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                train_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/train'
                val_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/validation'
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                train_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/train_folder'
                val_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/val_folder'
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                train_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/new/train1'
                val_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/new/validation'
            if not os.path.exists(train_path) or not os.path.exists(val_path):
                train_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/training'
                val_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/testing'
            if dataset == 'ACDC':
                # train_loader, val_loader = ACDCDataset('/home/hucon/vscode/u-kan/Seg_UKAN/inputs/archive/database/training/*', num=0.9, batch_size=2)
                root_dir = "/home/hucon/vscode/u-kan/Seg_UKAN/inputs/ACDC/"
                list_dir = "/home/hucon/vscode/u-kan/Seg_UKAN/inputs/ACDC/lists_ACDC"
                from torchvision import transforms as tt
                train_dataset = ACDC_dataset(root_dir, list_dir, split="train", transform=
                                   tt.Compose([RandomGenerator(output_size=[224,224])]))
                print("The length of train set is: {}".format(len(train_dataset)))
                train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
                db_val=ACDC_dataset(base_dir=root_dir, list_dir=list_dir, split="valid")
                val_loader=DataLoader(db_val, batch_size=1, shuffle=False)
                # db_test =ACDC_dataset(base_dir=args.volume_path,list_dir=args.list_dir, split="test")
                # testloader = DataLoader(db_test, batch_size=1, shuffle=False)
                # train_loader = torch.utils.data.DataLoader(
                #     train_data,
                #     batch_size=config['batch_size'],
                #     shuffle=True,
                #     num_workers=config['num_workers'],
                #     drop_last=False)
                # val_loader = torch.utils.data.DataLoader(
                #     test_data,
                #     batch_size=config['batch_size'],
                #     shuffle=False,
                #     num_workers=config['num_workers'],
                #     drop_last=False)
            else:
                print('train_path',train_path)
                print('val_path',val_path)
                train_dataset_path = f'{train_path}'+f'/images/*{img_ext}'
                val_dataset_path = f'{val_path}'+f'/images/*{img_ext}'
                train_img_ids = glob(train_dataset_path)
                val_img_ids = glob(val_dataset_path)
                print('train_img_ids',len(train_img_ids))
                print('val_img_ids',len(val_img_ids))
                train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
                val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
                
                shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
                shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')
                shutil.copy2('deform_lsk.py', f'{output_dir}/{exp_name}/')
                shutil.copy2('/home/hucon/vscode/u-kan/Seg_UKAN/msa2net_networks/msa2net_mx_y.py', f'{output_dir}/{exp_name}/')
                shutil.copy2('/home/hucon/vscode/u-kan/Seg_UKAN/denet.py', f'{output_dir}/{exp_name}/')

                val_dataset = Dataset(
                    img_ids=val_img_ids,
                    img_dir=val_path+'/images',
                    mask_dir=val_path+'/masks',
                    img_ext=img_ext,
                    mask_ext=mask_ext,
                    num_classes=config['num_classes'],
                    transform=val_transform)

                train_dataset = Dataset(
                    img_ids=train_img_ids,
                    img_dir=train_path+'/images',
                    mask_dir=train_path+'/masks',
                    img_ext=img_ext,
                    mask_ext=mask_ext,
                    num_classes=config['num_classes'],
                    transform=train_transform)

                train_loader = torch.utils.data.DataLoader(
                    train_dataset,
                    batch_size=config['batch_size'],
                    shuffle=True,
                    num_workers=config['num_workers'],
                    drop_last=False)
                val_loader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=config['batch_size'],
                    shuffle=False,
                    num_workers=config['num_workers'],
                    drop_last=False)
                ###########################
            log = OrderedDict([
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', []),
                ('val_loss', []),
                ('val_acc', []),
                ('val_iou', []),
                ('val_dice', []),
                ('best_val_iou',[]),
                ('hd95', []),
                ('recall', []),
                ('specificity', []),
                ('precision', [])
            ])
            start_epoch = 0
            if  continue_training == 1:
                model, optimizer, epoch,new_path = load_checkpoint(continue_path, model, optimizer)
                start_epoch  = epoch+1
                exp_name = new_path
                print(f"Loaded model from epoch {start_epoch}") 

            best_iou = 0
            best_dice= 0
            trigger = 0
            os.makedirs(f'{output_dir}/{exp_name}/epoch', exist_ok=True)


            for epoch in range(start_epoch, config['epochs']):
                print('Epoch [%d/%d]' % (epoch, config['epochs']))
                print(f'Iteration {iteration_count}: model_name {setting["model"]}')
                # train for one epoch
                train_log = train(config, train_loader, model, criterion, optimizer)
                # evaluate on validation set
                val_log = validate(config, val_loader, model, criterion)

                if config['scheduler'] == 'CosineAnnealingLR':
                    scheduler.step()
                elif config['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(val_log['loss'])

                print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                    % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
                log['epoch'].append(epoch)
                log['lr'].append(round(config['lr'], 5))
                log['loss'].append(round(train_log['loss'], 5))
                log['iou'].append(round(train_log['iou'], 5))
                log['val_loss'].append(round(val_log['loss'], 5))
                log['val_acc'].append(round(val_log['accuracy'], 5))
                log['val_iou'].append(round(val_log['iou'], 5))
                log['val_dice'].append(round(val_log['dice'], 5))
                log['best_val_iou'].append(round(best_iou, 5))
                log['hd95'].append(round(val_log['hd95'], 5))
                log['recall'].append(round(val_log['recall'], 5))
                log['specificity'].append(round(val_log['specificity'], 5))
                log['precision'].append(round(val_log['precision'], 5))
                if continue_training == 1:
                    pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/continue_log.csv', index=False)
                else:
                    pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

                my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
                my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
                my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
                my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
                my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

                my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
                my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)
                trigger += 1
                if (epoch+1) in [epoch]:
                    iou = val_log['iou']
                    save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                    # torch.save(model.state_dict(), f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                if val_log['iou'] > best_iou:
                    best_iou = val_log['iou']
                    best_dice = val_log['dice']
                    best_iou_epoch = epoch
                    best_model = model 
                    if config['dataset'] == 'GLAS_orin'and best_iou > 0.875:
                       save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                    if config['dataset'] == 'NEW_BUSI'and best_iou > 0.73:
                        save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                    # if best_iou > 0.86:
                    #     save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                    if config['dataset'] == 'CVC'and best_iou > 0.87:
                        save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                    if config['dataset'] == 'isic_201'and best_iou > 0.88:
                        save_checkpoint(epoch, model, optimizer, f'{output_dir}/{exp_name}/epoch/{epoch}_{best_iou}model.pth')
                    save_checkpoint(epoch, best_model, optimizer, f'{output_dir}/{exp_name}/epoch/best.pth')
                print(f"best model:IOU:{best_iou}, DICE:{best_dice},{best_iou_epoch}")
                #     # trigger = 0
                # # early stopping
                # if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                #     print("=> early stopping")
                #     break
            # torch.save(best_model, f'{output_dir}/{exp_name}/epoch/{epoch}_{best_iou}.pth')
            end = datetime.now().strftime("%Y-%m-%d %H:%M")
            # log.info('Best IoU: %.4f at epoch %d' % (best_iou, best_iou_epoch))
            print('end',end)
            print('Best IoU: %.4f at epoch %d' % (best_iou, best_iou_epoch))
    elif singal == 2:
            model = model.cuda()
            param_groups = []
            for name, param in model.named_parameters():
                if 'kan' in name.lower() and 'fc' in name.lower():
                    param_groups.append({'params': param, 'lr': config['kan_lr'], 'weight_decay': config['kan_weight_decay']}) 
                else:
                    param_groups.append({'params': param, 'lr': config['lr'], 'weight_decay': config['weight_decay']})  

            if config['optimizer'] == 'Adam':
                optimizer = optim.Adam(param_groups)


            elif config['optimizer'] == 'SGD':
                optimizer = optim.SGD(param_groups, lr=config['lr'], momentum=config['momentum'], nesterov=config['nesterov'], weight_decay=config['weight_decay'])
            else:
                raise NotImplementedError

            if config['scheduler'] == 'CosineAnnealingLR':
                scheduler = lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
            elif config['scheduler'] == 'ReduceLROnPlateau':
                scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'], verbose=1, min_lr=config['min_lr'])
            elif config['scheduler'] == 'MultiStepLR':
                scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')], gamma=config['gamma'])
            elif config['scheduler'] == 'ConstantLR':
                scheduler = None
            else:
                raise NotImplementedError
            dataset = config['dataset']
            train_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/train'
            val_path = f'/home/hucon/vscode/u-kan/Seg_UKAN/inputs/{dataset}/validation'
            from sklearn.model_selection import train_test_split


            print('train_path',train_path)
            print('val_path',val_path)
            train_dataset_path = f'{train_path}'+f'/images/*{img_ext}'
            val_dataset_path = f'{val_path}'+f'/images/*{img_ext}'
            train_img_ids = glob(train_dataset_path)

            val_img_ids = glob(val_dataset_path)

            train_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in train_img_ids]
            val_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_img_ids]
            train_ids, other_ids = train_test_split(train_img_ids, test_size=0.2, random_state=42)
            print('train_ids',len(train_ids))

            val_ids, test_ids = train_test_split(other_ids, test_size=0.5, random_state=42)
            print('val_ids',len(val_ids))
            print('test_ids',len(test_ids))

            train_ids_set = set(train_ids)
            val_ids_set = set(val_ids)
            test_ids_set = set(test_ids)

            # 检查训练集和验证集之间是否有重复的ID
            if len(train_ids_set.intersection(val_ids_set)) > 0:
                print('There are duplicate IDs between the training set and the validation set.')

            # 检查训练集和测试集之间是否有重复的ID
            if len(train_ids_set.intersection(test_ids_set)) > 0:
                print('There are duplicate IDs between the training set and the test set.')

            # 检查验证集和测试集之间是否有重复的ID
            if len(val_ids_set.intersection(test_ids_set)) > 0:
            ###################
                print('There are duplicate IDs between the validation set and the test set.')
            ######txt######
            txt_path = f'{output_dir}/{exp_name}'
            with open(f'{txt_path}/test_img.txt', 'w') as f:
                for img_id in test_ids:
                    f.write(f'{img_id}\n')
            with open(f'{txt_path}/val_img.txt', 'w') as f:
                for img_id in val_ids:
                    f.write(f'{img_id}\n')
            with open(f'{txt_path}/train_img.txt', 'w') as f:
                for img_id in train_ids:
                    f.write(f'{img_id}\n')
            ################
            shutil.copy2('train.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('archs.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('deform_lsk.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('/home/hucon/vscode/u-kan/Seg_UKAN/msa2net_networks/msa2net_mx_y.py', f'{output_dir}/{exp_name}/')
            shutil.copy2('/home/hucon/vscode/u-kan/Seg_UKAN/denet.py', f'{output_dir}/{exp_name}/')

            val_dataset = Dataset(
                img_ids=val_ids,
                img_dir=train_path+'/images',
                mask_dir=train_path+'/masks',
                img_ext=img_ext,
                mask_ext=mask_ext,
                num_classes=config['num_classes'],
                transform=val_transform)

            train_dataset = Dataset(
                img_ids=train_ids,
                img_dir=train_path+'/images',
                mask_dir=train_path+'/masks',
                img_ext=img_ext,
                mask_ext=mask_ext,
                num_classes=config['num_classes'],
                transform=train_transform)

            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config['batch_size'],
                shuffle=True,
                num_workers=config['num_workers'],
                drop_last=False)
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False)
            ###########################
            log = OrderedDict([
                ('epoch', []),
                ('lr', []),
                ('loss', []),
                ('iou', []),
                ('val_loss', []),
                ('val_iou', []),
                ('val_dice', []),
                ('best_val_iou',[]),
            ])


            best_iou = 0
            best_dice= 0
            trigger = 0
            os.makedirs(f'{output_dir}/{exp_name}/epoch', exist_ok=True)


            for epoch in range(config['epochs']):
                print('Epoch [%d/%d]' % (epoch, config['epochs']))

                # train for one epoch
                train_log = train(config, train_loader, model, criterion, optimizer)
                # evaluate on validation set
                val_log = validate(config, val_loader, model, criterion)

                if config['scheduler'] == 'CosineAnnealingLR':
                    scheduler.step()
                elif config['scheduler'] == 'ReduceLROnPlateau':
                    scheduler.step(val_log['loss'])

                print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
                    % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))

                log['epoch'].append(epoch)
                log['lr'].append(config['lr'])
                log['loss'].append(train_log['loss'])
                log['iou'].append(train_log['iou'])
                log['val_loss'].append(val_log['loss'])
                log['val_acc'].append(val_log['accuracy'])
                log['val_iou'].append(val_log['iou'])
                log['val_dice'].append(val_log['dice'])
                log['best_val_iou'].append(best_iou)
                pd.DataFrame(log).to_csv(f'{output_dir}/{exp_name}/log.csv', index=False)

                my_writer.add_scalar('train/loss', train_log['loss'], global_step=epoch)
                my_writer.add_scalar('train/iou', train_log['iou'], global_step=epoch)
                my_writer.add_scalar('val/loss', val_log['loss'], global_step=epoch)
                my_writer.add_scalar('val/iou', val_log['iou'], global_step=epoch)
                my_writer.add_scalar('val/dice', val_log['dice'], global_step=epoch)

                my_writer.add_scalar('val/best_iou_value', best_iou, global_step=epoch)
                my_writer.add_scalar('val/best_dice_value', best_dice, global_step=epoch)
                trigger += 1
                if (epoch+1)%50 == 0:
                    iou = val_log['iou']
                    torch.save(model.state_dict(), f'{output_dir}/{exp_name}/epoch/{epoch}_{iou}model.pth')
                if val_log['iou'] > best_iou:
                    best_iou = val_log['iou']
                    best_dice = val_log['dice']
                    best_iou_epoch = epoch
                    if config['dataset'] == 'GLAS_orin'and best_iou > 0.85:
                        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/{best_iou_epoch}_{best_iou}model.pth')
                    if config['dataset'] == 'NEW_BUSI'and best_iou > 0.68:
                        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/{best_iou_epoch}_{best_iou}model.pth')
                    if best_iou > 0.86:
                        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/{best_iou_epoch}_{best_iou}model.pth')
                    if config['dataset'] == 'CVC'and best_iou > 0.87:
                        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/{best_iou_epoch}_{best_iou}model.pth')
                    if config['dataset'] == 'isic_2016'and best_iou > 0.88:
                        torch.save(model.state_dict(), f'{output_dir}/{exp_name}/{best_iou_epoch}_{best_iou}model.pth')
                print(f"best model:IOU:{best_iou}, DICE:{best_dice},{best_iou_epoch}")
                # trigger = 0
                # early stopping
                if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
                    print("=> early stopping")
                    break

            end = datetime.now().strftime("%Y-%m-%d %H:%M")
            # log.info('Best IoU: %.4f at epoch %d' % (best_iou, best_iou_epoch))
            print('end',end)
            print('Best IoU: %.4f at epoch %d' % (best_iou, best_iou_epoch))
if __name__ == '__main__':
    def remove_last_two_parts(path):
        parts = path.rstrip(os.sep).split(os.sep)
        new_path = os.sep.join(parts[-4:-2])
        return new_path
    model_settings = [
        {'model':'Msa2Net_mx_y', 'signal': 0, 'batchsize': 4, 'lr': 0.0002, 'epoch': 200, 'data_name': 'cvc', 'keyword': 'loss_1_0', 'size': 224},
    }]
    for setting in model_settings:
        singal = setting['signal']
        main(setting['lr'], setting['batchsize'], setting['data_name'], setting['model'], setting['keyword'], setting['epoch'],setting['size'])
        print('model_name', setting['model'])
        iteration_count += 1
        if iteration_count==4:
            break
        print(f'Iteration {iteration_count}: model_name {setting["model"]}')

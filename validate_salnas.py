from salnas.losses.loss import *
import salnas.datasets.data as data
from argparse import ArgumentParser
from salnas.models.backbone import *
import torch
import torch.nn as nn
from salnas.utils.utils import AverageMeter2 as AverageMeter 
import salnas.utils.utils as utils
import time
import sys
from tqdm import tqdm
import warnings
from salnas.datasets.augment import *
import salnas.models
from salnas.config import setup

import logging
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="/home/mllab/proj/datasets/saliency/")
parser.add_argument('--input_size_h',default=224, type=int)
parser.add_argument('--input_size_w',default=224, type=int)
parser.add_argument('--no_workers',default=8, type=int)
parser.add_argument('--no_epochs',default=20, type=int)
parser.add_argument('--t_epochs',default=10, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_salicon_path',default="model_salnas_self.pt", type=str)
parser.add_argument('--output_dir', type=str, default="outputs")

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--student',default="salnas", type=str)
parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))
parser.add_argument('--seed',default=3407, type=int)

args = parser.parse_args()

torch.multiprocessing.freeze_support()
torch.cuda.empty_cache()
torch.cuda.reset_max_memory_allocated()

utils.fix_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

def model_load_state_dict(student, path_state_dict):
    # student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
    student.load_weights_from_pretrained_models_full(path_state_dict)
    print(path_state_dict,"loaded pre-trained student")

if args.dataset != "salicon":
    args.output_size = (384, 384)

args_super = setup("./configs/train_alphanet_models.yml")
student = salnas.models.create_model(args_super, output_size=args.output_size)
logging.info(args_super.resume)
student.load_weights_from_pretrained_models(args_super.resume)

logging.info(args.dataset)

train_loader, val_loader, output_size = data.create_dataset(args)

if args.dataset != "salicon":
    args.output_size = (384, 384)

model_load_state_dict(student, args.model_salicon_path)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    student = nn.DataParallel(student)

student.to(device)

def bn_calibration(model, loader, device):
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    model.eval()
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for (img, _, _) in tqdm(loader):
            img = img.to(device)
            _ = model(img) #forward only

def validate(subnet, val_loader,epoch, device):
    batch_time = AverageMeter('Time', ':6.3f')
    l_cc = AverageMeter('CC', ':6.2f')
    l_kld = AverageMeter('KLD', ':6.2f')
    l_nss = AverageMeter('NSS', ':6.2f')
    l_sim = AverageMeter('SIM', ':6.2f')
    l_auc = AverageMeter('AUC', ':6.2f')

    #evaluation
    end = time.time()

    bn_calibration(subnet, val_loader, device)

    subnet.eval()
    with torch.no_grad():
        for (img, gt, fixations) in tqdm(val_loader):
            img = img.to(device)
            gt = gt.to(device)
            fixations = fixations.to(device)

            # compute output
            output = subnet(img)

            cc_ = cc(output, gt)  
            kld_  = kldiv(output, gt) 
            nss_ = nss(output, fixations)
            sim_  = similarity(output, gt)   
            auc_ = auc_judd(output, fixations)   

            batch_size = img.size(0)

            l_cc.update(cc_, batch_size)
            l_kld.update(kld_, batch_size)
            l_nss.update(nss_, batch_size)
            l_sim.update(sim_, batch_size)
            l_auc.update(auc_, batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)

    print('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, AUC : {:.5f}  time:{:3f} minutes'.
          format(epoch, l_cc.avg, l_kld.avg, l_nss.avg, l_sim.avg, l_auc.avg , (time.time()-end)/60))
    sys.stdout.flush()

    nss_avg = ((torch.exp(l_nss.avg) / (1 + torch.exp(l_nss.avg))))
    metric_scores = torch.tensor([1-l_cc.avg, l_kld.avg, 1-nss_avg, 1-l_sim.avg, 1-l_auc.avg], dtype=torch.float32)
    return torch.sum(metric_scores)

if torch.cuda.device_count() > 1:
    student.module.sample_max_subnet()
    subnet = student.module.get_active_subnet(preserve_weight=True)
else: 
    student.sample_max_subnet()
    subnet = student.get_active_subnet(preserve_weight=True)

with torch.no_grad():
    print("Max subnet")
    cc_loss = validate(subnet, val_loader, 0, device)

if torch.cuda.device_count() > 1:
    student.module.sample_min_subnet()
    subnet = student.module.get_active_subnet(preserve_weight=True)
else: 
    student.sample_min_subnet()
    subnet = student.get_active_subnet(preserve_weight=True)

with torch.no_grad():
    print("Min subnet")
    cc_loss = validate(subnet, val_loader, 0, device)
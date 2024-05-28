import logging
import time
import sys
import os
import warnings

import torch
import torch.nn as nn
from argparse import ArgumentParser
from tqdm import tqdm
from salnas.utils.swa_utils import AveragedModel

# Import the configuration
import config

import salnas.datasets.data as data
from salnas.datasets.augment import *
from salnas.losses.loss import *
from salnas.models.backbone import *
import salnas.models
from salnas.config import setup
from salnas.utils.utils import AverageMeter2 as AverageMeter
import salnas.utils.utils as utils
from salnas.utils.carbon import CarbonAI

warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--dataset_dir", type=str, default=config.DATASET_PATH)
parser.add_argument('--no_workers',type=int, default=config.NO_WORKER)
parser.add_argument('--log_interval', type=int, default=config.LOS_INTERVAL)

parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument('--no_epochs',default=20, type=int)
parser.add_argument('--t_epochs',default=10, type=int)

parser.add_argument('--lr_sched',default=True, type=bool)
parser.add_argument('--model_val_path',default="model.pt", type=str)
parser.add_argument('--model_salicon_path',default="salnas30.pt", type=str)
parser.add_argument('--output_dir', type=str, default="outputs")

parser.add_argument('--loss_mode', type=str, default="new", choices=["pkd", "new"])

parser.add_argument('--kldiv', action='store_true', default=False)
parser.add_argument('--cc', action='store_true', default=False)
parser.add_argument('--nss', action='store_true', default=False)
parser.add_argument('--sim', action='store_true', default=False)
parser.add_argument('--l1', action='store_true', default=False)
parser.add_argument('--auc', action='store_true', default=False)

parser.add_argument('--kldiv_coeff',default=1.0, type=float)
parser.add_argument('--cc_coeff',default=-1.0, type=float)
parser.add_argument('--sim_coeff',default=-1.0, type=float)
parser.add_argument('--nss_coeff',default=1.0, type=float)
parser.add_argument('--l1_coeff',default=1.0, type=float)
parser.add_argument('--auc_coeff',default=1.0, type=float)

parser.add_argument('--amp', action='store_true', default=False)
parser.add_argument('--self_kd', action='store_true', default=False)
parser.add_argument('--alpha_self_kd',default=0.5, type=float)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--student',default="salnas", type=str)

parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))
parser.add_argument('--seed',default=3407, type=int)

args = parser.parse_args()

torch.multiprocessing.freeze_support()

scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

utils.fix_seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

csv_log = utils.OwnLogging()

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

args.save = '{}/{}-{}-{}'.format(args.output_dir, args.dataset, args.student, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=None)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

logging.getLogger('PIL').setLevel(logging.WARNING)

logging.info("Hyperparameter: %s" % args)
logging.info('-'*80)

start = time.time()

codecarbon = CarbonAI(country_iso_code="THA", args=args)
codecarbon.start()

def model_load_state_dict(student, path_state_dict):
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

if args.dataset != "salicon":
    model_load_state_dict(student, args.model_salicon_path)

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    student = nn.DataParallel(student)

student.to(device)

no_wd_params, wd_params = [], []
for name, param in student.named_parameters():
    if param.requires_grad:
        if ".bn" in name or ".bias" in name:
            no_wd_params.append(param)
        else:
            wd_params.append(param)
no_wd_params = nn.ParameterList(no_wd_params)
wd_params = nn.ParameterList(wd_params)

weight_decay_weight = 0.00001
weight_decay_bn_bias = 0.

params_group = [
    {"params": wd_params, "weight_decay": float(weight_decay_weight), 'group_name':'weight'},
    {"params": no_wd_params, "weight_decay": float(weight_decay_bn_bias), 'group_name':'bn_bias'},
]
optimizer = torch.optim.SGD(params_group, lr=args.lr, momentum=0.9, nesterov=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = args.t_epochs)

if args.self_kd:
    swa_model = AveragedModel(model=student, use_buffers=True, device=device)

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

    logging.info('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, AUC : {:.5f}  time:{:3f} minutes'.
          format(epoch, l_cc.avg, l_kld.avg, l_nss.avg, l_sim.avg, l_auc.avg , (time.time()-end)/60))
    sys.stdout.flush()

    nss_avg = ((torch.exp(l_nss.avg) / (1 + torch.exp(l_nss.avg))))
    metric_scores = torch.tensor([1-l_cc.avg, l_kld.avg, 1-nss_avg, 1-l_sim.avg, 1-l_auc.avg], dtype=torch.float32)
    csv_log.update(epoch, lr, l_cc.avg.item(), l_kld.avg.item(), l_nss.avg.item(), l_sim.avg.item(), l_auc.avg.item(), torch.sum(metric_scores).item())
    return torch.sum(metric_scores)

def train_ofa(model, optimizer, loader, epoch, device, args, args_super):
    model.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):

        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        num_subnet_training = max(2, getattr(args_super, 'num_arch_training', 2))
        optimizer.zero_grad()

        ### compute gradients using sandwich rule ###
        # step 1 sample the largest network, apply regularization to only the largest network
        if torch.cuda.device_count() > 1:
            model.module.sample_max_subnet()
            model.module.set_dropout_rate(0.2, 0.2, True) #dropout for supernet
        else: 
            model.sample_max_subnet()
            model.set_dropout_rate(0.2, 0.2, True) #dropout for supernet
            
        optimizer.zero_grad()

        if args.self_kd and epoch > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_map = swa_model(img)
                    soft_logits = pred_map.clone().detach()
                    gt = soft_logits + (gt - soft_logits) * args.alpha_self_kd

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map = model(img)
            assert pred_map.size() == gt.size()
            loss = loss_func(pred_map, gt, fixations, args)
        
        scaler.scale(loss).backward()

        #step 2. sample the smallest network and several random networks
        sandwich_rule = getattr(args_super, 'sandwich_rule', True)

        if torch.cuda.device_count() > 1:
            model.module.set_dropout_rate(0, 0, True)  #reset dropout rate
        else: 
            model.set_dropout_rate(0, 0, True)  #reset dropout rate

        for arch_id in range(1, num_subnet_training):
            if arch_id == num_subnet_training-1 and sandwich_rule:
                if torch.cuda.device_count() > 1:
                    model.module.sample_min_subnet()
                else: 
                    model.sample_min_subnet()
            else:
                if torch.cuda.device_count() > 1:
                    model.module.sample_active_subnet()
                else:
                    model.sample_active_subnet()

            with torch.cuda.amp.autocast(enabled=args.amp):
                pred_map_s = model(img)
                assert pred_map_s.size() == gt.size()
                loss = loss_func(pred_map_s, gt, fixations, args)

            scaler.scale(loss).backward()

        total_loss += loss.item()
        cur_loss += loss.item()
        
        scaler.unscale_(optimizer)
        #clip gradients if specfied
        torch.nn.utils.clip_grad_value_(model.parameters(), args_super.grad_clip_value)

        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60))
            cur_loss = 0.0
            sys.stdout.flush()

    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)

for epoch in range(0, args.no_epochs):
    # Print the learning rate at the end of each epoch
    lr = optimizer.param_groups[0]['lr']
    logging.info(f"Epoch {epoch}: Learning rate : {lr}")

    loss = train_ofa(student, optimizer , train_loader, epoch, device, args, args_super)

    if args.lr_sched:
        scheduler.step()

    if torch.cuda.device_count() > 1:
        student.module.sample_max_subnet()
        subnet = student.module.get_active_subnet(preserve_weight=True)
    else: 
        student.sample_max_subnet()
        subnet = student.get_active_subnet(preserve_weight=True)
    
    with torch.no_grad():
        cc_loss = validate(subnet, val_loader, epoch, device)
        if epoch == 0 :
            best_loss = cc_loss
        if best_loss >= cc_loss:
            best_loss = cc_loss
            logging.info('[{:2d},  save, {}]'.format(epoch, args.model_val_path))
            utils.save_state_dict(student, args)
            if args.self_kd:
                swa_model.update_parameters(student)
        print()

csv_log.done(os.path.join(args.save, 'val.csv'))

end = time.time()
logging.info("Time = %.2f Sec", (end - start))

codecarbon.stop()
logging.info("Emission = %.4f kgCO2" % codecarbon.emissions)
logging.info("Power = %.2f W" % codecarbon.power)
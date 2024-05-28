from salnas.losses.loss import *
import salnas.datasets.data as data
from argparse import ArgumentParser
from models.backbone import *
import torch
import torch.nn as nn
from salnas.utils.utils import AverageMeter
import time
import sys
from tqdm import tqdm
from ptflops import get_model_complexity_info
import kornia
import warnings
warnings.filterwarnings("ignore")

parser = ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--dataset_dir", type=str, default="/home/mllab/proj/datasets/saliency/")
parser.add_argument('--input_size_h',default=256, type=int)
parser.add_argument('--input_size_w',default=256, type=int)
parser.add_argument('--no_workers',default=16, type=int)
parser.add_argument('--log_interval',default=20, type=int)
parser.add_argument('--pretrained',default="salnas-tmp-NewLossMMAd20EMA.pt", type=str)

parser.add_argument('--dataset',default="salicon", type=str)
parser.add_argument('--student',default="salnas", type=str)
parser.add_argument('--teacher',default="ofa595", type=str)

parser.add_argument('--readout',default="simple", type=str)
parser.add_argument('--output_size', default=(480, 640))

parser.add_argument('--mode',default="nkd", type=str)
parser.add_argument('--mixed',default=True, type=bool)
parser.add_argument('--seed',default=3407, type=int)

args = parser.parse_args()

def model_load_state_dict(student , teacher, path_state_dict):
    if args.mode == "kd":
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        teacher.load_state_dict(torch.load(path_state_dict)["teacher"], strict=True)
        print("loaded pre-trained student and teacher")
    else: 
        student.load_state_dict(torch.load(path_state_dict)["student"], strict=True)
        print("loaded pre-trained student")

if args.dataset != "salicon":
    args.output_size = (384, 384)

if args.student == "eeeac2":
    student = EEEAC2(num_channels=3, train_enc=True, load_weight=True, output_size=args.output_size, readout=args.readout)
elif args.student == "eeeac1":
    student = EEEAC1(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "mbv2":
    student = MobileNetV2(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "mbv3":
    student = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "efb0":
    student = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "efb4":
    student = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "efb7":
    student = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "ghostnet":
    student = GhostNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "rest":
    student = ResT(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.student == "vgg":
    student = VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
elif args.student == "salnas":
    import salnas.models
    from salnas.config import setup
    args_super = setup("./configs/train_alphanet_models.yml")
    student = salnas.models.create_model(args_super, output_size=args.output_size)
    print(args_super.resume)
    student.load_weights_from_pretrained_models(args_super.resume)


torch.multiprocessing.freeze_support()

print(args.dataset)

train_loader, val_loader, output_size = data.create_dataset(args)

if args.dataset != "salicon":
    args.output_size = (384, 384)

if args.teacher == "ofa595":
    teacher = OFA595(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "tresnet":
    teacher = tresnet(num_channels=3, train_enc=True, load_weight=1, pretrained='1k', output_size=args.output_size)
elif args.teacher == "mbv3":
    teacher = MobileNetV3_1k(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "efb0":
    teacher = EfficientNet(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "efb4":
    teacher = EfficientNetB4(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "efb7":
    teacher = EfficientNetB7(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size, readout=args.readout)
elif args.teacher == "pnas":
    teacher = PNASModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)
elif args.teacher == "vgg":
    teacher = VGGModel(num_channels=3, train_enc=True, load_weight=1, output_size=args.output_size)

model_load_state_dict(student , teacher, args.pretrained)


if torch.cuda.device_count() > 1:
    student.module.sample_max_subnet()
    student = student.module.get_active_subnet(preserve_weight=True)
else: 
    student.sample_max_subnet()
    student = student.get_active_subnet(preserve_weight=True)

print("Teacher:")
macs, params = get_model_complexity_info(teacher, (3, args.input_size[0], args.input_size[1]), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

print("Student:")
macs, params = get_model_complexity_info(student, (3, args.input_size[0], args.input_size[1]), as_strings=True,
                                        print_per_layer_stat=False, verbose=False)
print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    student = nn.DataParallel(student)
    if args.mode == "kd":
        teacher = nn.DataParallel(teacher)

student.to(device)

if args.mode == "kd":
    teacher.to(device)
else: 
    teacher = None

def bn_calibration(model, loader, device):
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    model.eval()
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for (img, img_id, sz) in tqdm(loader):
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

with torch.no_grad():
    if args.mode == "kd":
        # print("Teacher:")
        # _ = validate(teacher, val_loader, device)
        print("Student:")
        cc_loss = validate(student, val_loader, 0, device)
    else :
        cc_loss = validate(student, val_loader, 0, device)
import torch
from tqdm import tqdm
import time
import sys
from salnas.utils.utils import AverageMeter
from salnas.losses.loss import cc, kldiv, nss, similarity, auc_judd
import logging

def bn_calibration(model, loader, device):
    # bn running stats calibration following Slimmable (https://arxiv.org/abs/1903.05134)
    # please consider trying a different random seed if you see a small accuracy drop
    model.eval()
    with torch.no_grad():
        model.reset_running_stats_for_calibration()
        for (img, _, _) in tqdm(loader):
            img = img.to(device)
            _ = model(img) #forward only

def validate(model, loader, epoch, device, csv_log, lr, bn_calibration=False):
    model.eval()
    tic = time.time()
    cc_loss = AverageMeter()
    kldiv_loss = AverageMeter()
    nss_loss = AverageMeter()
    sim_loss = AverageMeter()
    auc_loss = AverageMeter()

    if bn_calibration:
        bn_calibration(model, loader, device)
    
    for (img, gt, fixations) in tqdm(loader):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)
        
        pred_map = model(img)

        cc_loss.update(cc(pred_map, gt))    
        kldiv_loss.update(kldiv(pred_map, gt))    
        nss_loss.update(nss(pred_map, fixations))    
        sim_loss.update(similarity(pred_map, gt))    
        auc_loss.update(auc_judd(pred_map, fixations))    

    logging.info('[{:2d},   val] CC : {:.5f}, KLDIV : {:.5f}, NSS : {:.5f}, SIM : {:.5f}, AUC : {:.5f}  time:{:3f} minutes'.format(epoch, cc_loss.avg, kldiv_loss.avg, nss_loss.avg, sim_loss.avg, auc_loss.avg, (time.time()-tic)/60))
    sys.stdout.flush()
    
    nss_avg = ((torch.exp(nss_loss.avg) / (1 + torch.exp(nss_loss.avg))))
    metric_scores = torch.tensor([1-cc_loss.avg, kldiv_loss.avg, 1-nss_avg, 1-sim_loss.avg, 1-auc_loss.avg], dtype=torch.float32)
    
    if csv_log is not None:
        csv_log.update(epoch, lr, cc_loss.avg.item(), kldiv_loss.avg.item(), nss_loss.avg.item(), sim_loss.avg.item(), auc_loss.avg.item(), torch.sum(metric_scores).item())
    
    return torch.sum(metric_scores)
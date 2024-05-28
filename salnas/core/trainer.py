import torch
import time
import sys
import logging
from tqdm import tqdm
from salnas.losses.loss import loss_func

def train_baseline(student, optimizer, loader, epoch, device, args, scaler):
    student.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, gt, fixations, args)

        scaler.scale(loss_s).backward()
        total_loss += loss_s.item()
        cur_loss += loss_s.item()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)


def train_pkd(student, optimizer, loader, epoch, device, args, teacher, scaler):
    student.train()
    teacher.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map = teacher(img)
            loss = loss_func(pred_map, gt, fixations, args)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, pred_map.detach(), fixations, args)

        scaler.scale(loss + loss_s).backward()
        total_loss += loss.item() + loss_s.item()
        cur_loss += loss.item() + loss_s.item()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)


def train_self_kd(student, optimizer, loader, epoch, device, args, swa_model, scaler):
    student.train()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0

    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        if args.kd_mode == 'self' and epoch > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_map = swa_model(img)
                    soft_logits = pred_map.clone().detach()
                    gt = soft_logits + (gt - soft_logits) * 0.5
                    loss = loss_func(pred_map, gt, fixations, args)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, gt, fixations, args)

        if args.kd_mode == 'self' and epoch > 0:
            scaler.scale(loss + loss_s).backward()
            total_loss += loss.item() + loss_s.item()
            cur_loss += loss.item() + loss_s.item()
        else:
            scaler.scale(loss_s).backward()
            total_loss += loss_s.item()
            cur_loss += loss_s.item()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)



def train_ps_kd(student, optimizer, loader, epoch, device, args, teacher, scaler, alpha):
    student.train()
    teacher.eval()
    tic = time.time()
    
    total_loss = 0.0
    cur_loss = 0.0
    
    for idx, (img, gt, fixations) in enumerate(tqdm(loader)):
        img = img.to(device)
        gt = gt.to(device)
        fixations = fixations.to(device)

        optimizer.zero_grad()

        if args.kd_mode == 'ps' and epoch > 0:
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=args.amp):
                    pred_map = teacher(img)
                    soft_logits = pred_map.clone().detach()
                    #gt = soft_logits + (gt - soft_logits) * 0.5
                    # update gt by PS-KD 
                    gt = (alpha * soft_logits) + ((1-alpha) * gt)
                    loss = loss_func(pred_map, gt, fixations, args)

        with torch.cuda.amp.autocast(enabled=args.amp):
            pred_map_student = student(img)
            loss_s = loss_func(pred_map_student, gt, fixations, args)

        if args.kd_mode == 'ps' and epoch > 0:
            scaler.scale(loss + loss_s).backward()
            total_loss += loss.item() + loss_s.item()
            cur_loss += loss.item() + loss_s.item()
        else:
            scaler.scale(loss_s).backward()
            total_loss += loss_s.item()
            cur_loss += loss_s.item()

        # Unscales the gradients of optimizer's assigned params in-place
        scaler.unscale_(optimizer)

        # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
        torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)

        scaler.step(optimizer)
        scaler.update()

        if idx%args.log_interval==(args.log_interval-1):
            logging.info('[{:2d}, {:5d}] avg_loss : {:.5f}, time:{:3f} minutes, GPU Mem: {:.2f} MB'.format(epoch, idx, cur_loss/args.log_interval, (time.time()-tic)/60, round(torch.cuda.max_memory_allocated() / 1024.0 / 1024.0)))
            cur_loss = 0.0
            sys.stdout.flush()
    
    logging.info('[{:2d}, train] avg_loss : {:.5f}'.format(epoch, total_loss/len(loader)))
    sys.stdout.flush()

    return total_loss/len(loader)

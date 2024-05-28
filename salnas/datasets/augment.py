import torch
from salnas.losses.loss import *
import numpy as np

def fgsm(image, label, fixations, model, args, device, scaler):
    """
    Performs the fast gradient sign method for adversarial training
    """
    eps = 0.01
    image = image.to(device)
    label = label.to(device)
    fixations = fixations.to(device)
    image.requires_grad = True

    model.zero_grad()
    with torch.cuda.amp.autocast(enabled=args.amp):
        output = model(image)
        loss = loss_func(output, label, fixations, args)
        scaler.scale(loss).backward()

    gradient = image.grad.data
    sign_data = gradient.sign()
    adversarial_example = image + eps * sign_data
    adversarial_example = torch.clamp(adversarial_example, 0, 1)
    return adversarial_example

def mixup(img, gt):
    lam = np.random.beta(0.4, 0.4)
    indices = torch.randperm(img.size(0))
    shuffled_images = img[indices]
    shuffled_masks = gt[indices]
    img = lam * img + (1 - lam) * shuffled_images
    gt = lam * gt + (1 - lam) * shuffled_masks
    return img, gt
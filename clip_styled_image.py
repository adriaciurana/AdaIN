import os
import argparse
from PIL import Image
import torch
import datetime
import numpy as np
import torchvision
from params import Params
from torchvision import transforms
from torch.nn import functional as F

import clip
from model import AdaINModel
from utils import (
    normalize as normalize_adain,
    unnormalize as unnormalize_adain,
    compute_mu_sigma as compute_mu_sigma_adain,
    norm_by_mu_sigma as norm_by_mu_sigma_adain,
    apply_mu_sigma as apply_mu_sigma_adain
)
from logger import Logger

IM_SIZE = 224
def compute_text_direction(source_features, target_features):
    text_direction = (target_features - source_features).mean(axis=0, keepdim=True)
    text_direction /= text_direction.norm(dim=-1, keepdim=True)
    return text_direction

def clip_directional_loss(image_direction_features, text_direction_features):
    return (1. - F.cosine_similarity(image_direction_features, text_direction_features)).mean()

    # return 1e3 * F.mse_loss(image_features, text_features)    
    # return 1 - (image_features @ text_features.t()).sum()
    # return image_features.sub(text_features).norm(dim=1).div(2).arcsin().pow(2).mul(2)

def clip_loss(image_features, text_features):
    return (1. - F.cosine_similarity(image_features, text_features)).mean()

def content_image_loss(g_t, im):
    im_m = im.mean(dim=1)
    g_t_m = g_t.mean(dim=1)
    im_dx = torch.diff(im_m, dim=2)
    im_dy = torch.diff(im_m, dim=1)
    gt_dx = torch.diff(g_t_m, dim=2)
    gt_dy = torch.diff(g_t_m, dim=1)

    diff_loss = F.mse_loss(im_dx, gt_dx) + F.mse_loss(im_dy, gt_dy)
    color_loss = F.mse_loss(im, g_t)

    return diff_loss + 0.01 * color_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train algorithm')
    parser.add_argument('-w', '--weights_path', type=str, help='Weights of the AdaIN model', required=True)
    parser.add_argument('-d', '--device', type=str, help='Device for the inference', default=0)
    parser.add_argument('-i', '--image_path', type=str, help='Path of the image', required=True)
    parser.add_argument('-o', '--output_path', type=str, help='Image output', default="./clip_style", required=False)
    parser.add_argument('-a', '--alpha', type=float, help='Alpha', required=False, default=1.)
    parser.add_argument('-ta', '--target_text', type=str, help='Text class target', required=True)
    parser.add_argument('-ts', '--source_text', type=str, help='Text class source', required=False)
    parser.add_argument('-l', '--loss_type', type=str, help='Loss type', default='normal')
    args = parser.parse_args()
    
    DEVICE = torch.device("cuda:%d" % (args.device, ) if torch.cuda.is_available() else "cpu")
    OUTPUT_FOLDER = os.path.join(args.output_path, datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    """
        CLIP
    """
    normalize_images_clip = torchvision.transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    model_clip, preprocess = clip.load("ViT-B/32", device=DEVICE) #"RN50", device=DEVICE)
    model_clip = model_clip.eval().requires_grad_(False).float()

    """
        AdaIN Model
    """
    transform_adain = transforms.Compose([
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.CenterCrop(IM_SIZE),
        transforms.ToTensor(),
        normalize_adain,
    ])

    model_adain = AdaINModel()
    model_adain.load_state_dict(torch.load(args.weights_path), strict=True)
    model_adain = model_adain.to(DEVICE)
    model_adain = model_adain.eval().requires_grad_(False).float()

    """
        Image
    """
    # Preprocess image
    im_pil = Image.open(args.image_path)
    im = transform_adain(im_pil).unsqueeze(0).to(device=DEVICE).requires_grad_(False)

    # Get AdaIN features
    with torch.no_grad():
        _, feat_content = model_adain.encoder(im)
        mu_content, sigma_content = compute_mu_sigma_adain(feat_content)
        feat_norm_content = norm_by_mu_sigma_adain(feat_content, mu_content, sigma_content)

    """
        Text
    """
    # Get text features
    with torch.no_grad():
        if args.loss_type == 'directional':
            source_text = clip.tokenize([args.source_text]).to(DEVICE)
            source_features = model_clip.encode_text(source_text)
            source_features = source_features.float() / source_features.norm(dim=-1, keepdim=True)

        target_text = clip.tokenize([args.target_text]).to(DEVICE)
        target_features = model_clip.encode_text(target_text)
        target_features = target_features.float() / target_features.norm(dim=-1, keepdim=True)

    if args.loss_type == 'directional':
        text_direction = compute_text_direction(source_features, target_features)

        im_clip = normalize_images_clip(torch.clip(unnormalize_adain(im), 0., 1.))
        image_features = model_clip.encode_image(im_clip)
        image_features_orig = model_clip.encode_image(im_clip)
        image_features_orig = image_features_orig / image_features_orig.norm(dim=-1, keepdim=True)

    """
        Optimization problem
    """
    os.makedirs(os.path.join(OUTPUT_FOLDER, "steps"), exist_ok=True)

    OUTPUT_LOGS = os.path.join(OUTPUT_FOLDER, 'logs')
    os.makedirs(OUTPUT_LOGS, exist_ok=True)
    logger = Logger(
        Params.LOGGER_PATTERN,
        OUTPUT_LOGS
    )

    mu_style = torch.empty(mu_content.shape, dtype=torch.float, device=DEVICE, requires_grad=True)
    torch.nn.init.zeros_(mu_style)

    sigma_style = torch.empty(sigma_content.shape, dtype=torch.float, device=DEVICE, requires_grad=True)
    torch.nn.init.ones_(sigma_style)
    # mu_style = mu_content.clone().requires_grad_(True)
    # sigma_style = sigma_content.clone().requires_grad_(True)

    opt = torch.optim.AdamW(lr=1e-2, params=[mu_style, sigma_style])
    pbar = logger.create_train_bar(range(100_000), 0)

    for i in pbar:
        opt.zero_grad()

        # Apply style and generate
        t = apply_mu_sigma_adain(feat_norm_content, mu_style, sigma_style)
        interpolate_t = args.alpha * t + (1. - args.alpha) * feat_content
        g_t = model_adain.decoder(interpolate_t)

        # Get clip features (not necessary unnorm (adain) and norm (clip) because both uses imagenet norm)
        im_clip = normalize_images_clip(torch.clip(unnormalize_adain(g_t), 0., 1.))
        image_features = model_clip.encode_image(im_clip)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Loss
        losses = {}

        if args.loss_type == 'directional':
            edit_direction = (image_features - image_features_orig)
            edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True)
            losses['clip_direction_class'] = clip_directional_loss(edit_direction, text_direction)
        
        elif args.loss_type == 'normal':
            losses['clip_class'] = clip_loss(image_features, target_features)
        
        losses['content_image_loss'] = 10 * content_image_loss(g_t, im)

        loss = 0.
        for v in losses.values():
            loss += v
        losses['loss'] = loss

        loss.backward()
        opt.step()

        for k in losses.keys():
            losses[k] = losses[k].cpu().detach().numpy()

        logger.show_bar_train(pbar, 0, losses, metrics={})
        with torch.no_grad():
            if i % 1000 == 0:
                im_np = np.uint8(255 * torch.clip(unnormalize_adain(g_t), 0., 1.).permute(0, 2, 3, 1).cpu().numpy()[0])
                Image.fromarray(im_np).save(os.path.join(OUTPUT_FOLDER, 'steps', f'step_{i}.jpg'))

    
    with torch.no_grad():
        im_np = np.uint8(255 * torch.clip(unnormalize_adain(g_t), 0., 1.).permute(0, 2, 3, 1).cpu().numpy()[0])
        Image.fromarray(im_np).save(os.path.join(OUTPUT_FOLDER, f'final.jpg'))
            
            
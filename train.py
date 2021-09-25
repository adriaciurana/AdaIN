from __future__ import print_function
from __future__ import division

import os
import gc
import random
import glob
import argparse
import datetime
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from params import Params
from dataset import StyleTransferDataset
from model import AdaINModel
from losses import content_loss, style_loss
from utils import NaiveWith, adaIN, create_mosaic

from logger import Logger


def train(dataloader, model, optimizer, epoch, logger, mixed_precision):
    if mixed_precision and DEVICE != 'cpu':
        autocast = torch.cuda.amp.autocast()
        scaler = torch.cuda.amp.GradScaler()

    else:
        autocast = NaiveWith()
        scaler = None

    logger.clear()
    model.train()
    pbar = logger.create_train_bar(dataloader, epoch)
    for batch_i, (images_content, images_style) in enumerate(pbar):
        images_content = images_content.to(DEVICE)
        images_style = images_style.to(DEVICE)

        # Optimizer zero
        optimizer.zero_grad()

        # Model inf
        with autocast:
            _, feat_content = model.encoder(images_content)
            phi_list_style, feat_style = model.encoder(images_style)

            # AdaIN
            t = adaIN(feat_content, feat_style)
            g_t = model.decoder(t)
            phi_list_g_t, feat_g_t = model.encoder(g_t)

            # Loss content
            loss_content = content_loss(feat_g_t, feat_content)

            # Loss style
            loss_style = style_loss(phi_list_g_t, phi_list_style)
            
            # Loss
            losses = {}
            losses['content'] = Params.WEIGHTS['content'] * loss_content / TOTAL_WEIGHTS
            losses['style'] = Params.WEIGHTS['style'] * loss_style / TOTAL_WEIGHTS

            loss = 0.
            for v in losses.values():
                loss += v
            losses['loss'] = loss

            if torch.isnan(loss):
                continue

        for k in losses.keys():
            losses[k] = losses[k].cpu().detach().numpy()
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        del loss

        # Optimizer
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # Clean memory
        del images_content, images_style

        # logging
        if args.local_rank == 0:
            logger.show_bar_train(pbar, epoch, losses, metrics={})

    # save epoch
    if args.local_rank == 0:
        logger.write_train_epoch(pbar, epoch)


def test(dataloader, model, epoch, logger, mixed_precision):
    if mixed_precision and DEVICE != 'cpu':
        autocast = torch.cuda.amp.autocast()

    else:
        autocast = NaiveWith()

    logger.clear()
    model.eval()
    pbar = logger.create_test_bar(dataloader, epoch)
    with torch.no_grad():
        for batch_i, (images_content, images_style) in enumerate(pbar):
            images_content = images_content.to(DEVICE)
            images_style = images_style.to(DEVICE)

            # Model inf
            with autocast:
                _, feat_content = model.encoder(images_content)
                phi_list_style, feat_style = model.encoder(images_style)

                # AdaIN
                t = adaIN(feat_content, feat_style)
                g_t = model.decoder(t)
                phi_list_g_t, feat_g_t = model.encoder(g_t)

                # Loss content
                loss_content = content_loss(feat_g_t, feat_content)

                # Loss style
                loss_style = style_loss(phi_list_g_t, phi_list_style)
                
                # Loss
                losses = {}
                losses['content'] = Params.WEIGHTS['content'] * loss_content / TOTAL_WEIGHTS
                losses['style'] = Params.WEIGHTS['style'] * loss_style / TOTAL_WEIGHTS

                loss = 0.
                for v in losses.values():
                    loss += v
                losses['loss'] = loss

                if torch.isnan(loss):
                    continue

            for k in losses.keys():
                losses[k] = losses[k].cpu().detach().numpy()

            # Clean memory
            del images_content, images_style

            # logging
            if args.local_rank == 0:
                logger.show_bar_test(pbar, epoch, losses, metrics={})

    # save epoch
    if args.local_rank == 0:
        logger.write_test_epoch(pbar, epoch)
        
        # Create mosaic:
        (
            mosaic_content,
            mosaic_style,
            mosaic_content_styled
        ) = create_mosaic(dataloader, model, device=DEVICE)

        logger.get_test_tensorboard().add_image(
            'StyleTransferExamples: Content', 
            mosaic_content, 
            epoch
        )
        logger.get_test_tensorboard().add_image(
            'StyleTransferExamples: Style', 
            mosaic_style, 
            epoch
        )
        logger.get_test_tensorboard().add_image(
            'StyleTransferExamples: Content-Style', 
            mosaic_content_styled, 
            epoch
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train algorithm')
    parser.add_argument('-e', '--experiment', type=str, help='Name of the experiment', required=True)
    parser.add_argument('-b', '--backbone', type=str, help='Backbone inside the architecture', default='drn')
    parser.add_argument('-rc', '--resume_check', type=str, help='Path to checkpoint to resume training', default=None)
    parser.add_argument('-ro', '--resume_opt', type=str, help='Path to optimizer to resume training', default=None)
    parser.add_argument('-d', '--device', type=str, help='Device for the training', default=0)
    parser.add_argument('-m', '--mixed_precision', help='Enable mixed precision', action='store_true')
    parser.add_argument('--local_rank', type=int, default=0) # Currently USELESS
    args = parser.parse_args()

    # define Device
    DEVICE = torch.device("cuda:%d" % (args.device, ) if torch.cuda.is_available() else "cpu")
    TOTAL_WEIGHTS = sum(Params.WEIGHTS.values())

    # Create project folder and names
    OUTPUT_FOLDER = os.path.join(Params.OUTPUT, args.experiment, datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S"))
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    OUTPUT_LOGS = os.path.join(OUTPUT_FOLDER, 'logs')
    os.makedirs(OUTPUT_LOGS, exist_ok=True)
    logger = Logger(
        Params.LOGGER_PATTERN,
        OUTPUT_LOGS
    )

    OUTPUT_CHECKPOINTS = os.path.join(OUTPUT_FOLDER, 'checkpoints')
    os.makedirs(OUTPUT_CHECKPOINTS, exist_ok=True)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    content_folder_path = Params.DATASET_PATH['content']
    style_folder_path = Params.DATASET_PATH['style']

    content_images_path = glob.glob(os.path.join(content_folder_path, "**", "*.jpg"), recursive=True)
    random.shuffle(content_images_path)
    style_images_path = glob.glob(os.path.join(style_folder_path, "**", "*.jpg"), recursive=True)
    random.shuffle(style_images_path)

    num_val_content = int(Params.VAL_SPLIT * len(content_images_path))
    num_val_style = int(Params.VAL_SPLIT * len(style_images_path))

    val_content_images_path = content_images_path[:num_val_content]
    val_style_images_path = style_images_path[:num_val_style]

    train_content_images_path = content_images_path[num_val_content:]
    train_style_images_path = style_images_path[num_val_style:]

    train_dataset = StyleTransferDataset(
        content_images_path=train_content_images_path,
        style_images_path=train_style_images_path,
        transform=train_transform,
        n=10000
    )
    test_dataset = StyleTransferDataset(
        content_images_path=val_content_images_path,
        style_images_path=val_style_images_path,
        transform=val_transform,
        n=100
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Params.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed()
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Params.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        worker_init_fn=lambda x: np.random.seed()
    )


    # Model
    model = AdaINModel()

    if args.resume_check is not None:
        weights_dict = torch.load(args.resume_check)
        model_dict = model.state_dict()

        # Only load weights that are in the current model
        weights_dict = {k: v for k, v in weights_dict.items() if k in model_dict}
        model_dict.update(weights_dict)
        model.load_state_dict(model_dict, strict=False)

    # All to this device
    model = model.to(DEVICE)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=Params.LR, eps = 1e-5)

    # Epochs
    for epoch in range(Params.EPOCHS):
        train(train_dataloader, model, optimizer, epoch, logger, args.mixed_precision)
        gc.collect()
        test(test_dataloader, model, epoch, logger, args.mixed_precision)
        gc.collect()

        if epoch > 0 and epoch % Params.CHECKPOINT_SAVE_INTERVAL == 0 and args.local_rank == 0:
            print('Saved checkpoint')
            model_path_solve = os.path.join(OUTPUT_CHECKPOINTS, Params.MODEL_PATTERN.format(epoch=epoch, loss=logger.losses_avg['VAL']['loss'].avg))
            optimizer_path_solve = os.path.join(OUTPUT_CHECKPOINTS, model_path_solve.replace('checkpoint_', 'optim_'))
            torch.save(model.state_dict(), model_path_solve)
            torch.save(optimizer.state_dict(), optimizer_path_solve)

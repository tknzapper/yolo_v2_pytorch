import argparse
import os
import re
from torch.utils.tensorboard.writer import SummaryWriter
from voc2007 import VOCDataset, detection_collate
from torch.utils.data import DataLoader
from model import Yolo_v2
from loss import Loss
import config as cfg
import torch
import torch.nn as nn
import numpy as np
import time
import albumentations as A


# torch.manual_seed(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v2')
    parser.add_argument('--max_epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=cfg.epochs, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        default=1, type=int)
    parser.add_argument('--dataset', dest='dataset',
                        default='test', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=8, type=int)
    parser.add_argument('--output_dir', dest='output_dir',
                        default=cfg.output_dir, type=str)
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        default=True, type=bool)
    parser.add_argument('--display_interval', dest='display_interval',
                        default=10, type=int)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        default=False, type=bool)
    parser.add_argument('--save_interval', dest='save_interval',
                        default=20, type=int)
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--resume', dest='resume',
                        default=False, type=bool)
    parser.add_argument('--checkpoint_epoch', dest='checkpoint_epoch',
                        default=160, type=int)
    parser.add_argument('--exp_name', dest='exp_name',
                        default='default', type=str)

    args = parser.parse_args()
    return args

def config(resize):
    path = './config.py'
    p = re.compile(r'resize = [(]\d*[,]\s\d*[)]')
    with open(path, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if p.match(line):
                f.write(p.sub(f'resize = {resize}', line))
            else:
                f.write(line)
        f.truncate()

def start_train():

    # define hyper parameters
    args = parse_args()
    args.lr = cfg.lr
    args.milestone = cfg.lr_decay
    args.batch_size = cfg.batch_size
    args.weights_file = os.path.join('weights', 'pretrained', 'darknet19.pth')

    print('Called with args:')
    print(args)
    lr = args.lr

    # initiate tensorboard
    if args.use_tfboard:
        if args.exp_name == 'default':
            writer = SummaryWriter()
        else:
            writer = SummaryWriter('runs/' + args.exp_name)

    if args.dataset == 'voc07':
        root = os.path.join(cfg.data_root, 'Images/Train/')

    elif args.dataset == 'aa':
        root = os.path.join(cfg.data_root, cfg.img_root)
        args.batch_size = 5

    elif args.dataset == 'test':
        root = os.path.join(cfg.data_root, 'Images/Test/')
        args.batch_size = 11

    else:
        raise NotImplementedError

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    # define transform
    transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        # A.Resize(cfg.resize[0], cfg.resize[1], p=1),
        A.Normalize(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # load dataset
    print('loading dataset....')
    train_dataset = VOCDataset(img_root=root, transform=transform)

    print('dataset loaded.')

    print('training rois number: {}'.format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers,
                                  collate_fn=detection_collate, drop_last=True)

    # initialize model
    print('initialize the model')
    tic = time.time()
    model = Yolo_v2(weights_file=args.weights_file).to(device)
    if args.mGPUs:
        model = nn.DataParallel(model)
    toc = time.time()
    print('model loaded: cost time {:.2f}s'.format(toc-tic))

    # initialize criterion, optimizer, scheduler
    criterion = Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone, gamma=0.1, verbose=True)
    iters_per_epoch = int(len(train_dataset) / args.batch_size)

    if args.resume:
        print('resume training enable')
        resume_checkpoint_name = 'yolov2_epoch_{}.pth'.format(args.checkpoint_epoch)
        resume_checkpoint_path = os.path.join(output_dir, resume_checkpoint_name)
        print('resume from {}'.format(resume_checkpoint_path))
        checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        args.start_epoch = checkpoint['epoch'] + 1
        lr = checkpoint['lr']
        print('learning rate is {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # start training
    for epoch in range(args.start_epoch, args.max_epochs+1):
        if cfg.multi_scale and epoch in cfg.epoch_scale:
            cfg.scale_range = cfg.epoch_scale[epoch]
            print(f'change scale range to {cfg.scale_range}')

        loss_temp = 0
        tic = time.time()
        for step, (inputs, targets) in enumerate(train_loader):
            if cfg.multi_scale and (step + 1) % cfg.scale_step == 0:
                scale_index = np.random.randint(*cfg.scale_range)
                resize = cfg.input_sizes[scale_index]
                config(resize)
                print(f'change input size {resize}')

            inputs = inputs.to(device)
            targets = targets.to(device).float()
            ouputs = model(inputs)

            box_loss, conf_loss, noobj_loss, cls_loss = criterion(ouputs, targets)
            loss = box_loss + conf_loss + noobj_loss + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_temp += loss.item()
            if (step + 1) % args.display_interval == 0:
                toc = time.time()
                loss_temp /= args.display_interval

                print("[epoch {:2d}][step {:3d}/{:3d}] loss: {:.4f}, time cost {:.1f}s"
                      .format(epoch, step+1, iters_per_epoch, loss, toc-tic))
                # print(f"[epoch {epoch:2d}][step {step+1:3d}/{iters_per_epoch:3d}] loss: {loss:.4f}")
                print(f"\t\tbox_loss: {box_loss:.4f}, conf_loss: {conf_loss:.4f}, "
                      f"noobj_loss: {noobj_loss:.4f}, cls_loss: {cls_loss:.4f}")
                print()

                if args.use_cuda:
                    n_iter = (epoch - 1) * iters_per_epoch + step + 1
                    writer.add_scalar("losses/loss", loss_temp, n_iter)
                    writer.add_scalar("losses/box_loss", box_loss, n_iter)
                    writer.add_scalar("losses/conf_loss", conf_loss, n_iter)
                    writer.add_scalar("losses/noobj_loss", noobj_loss, n_iter)
                    writer.add_scalar("losses/cls_loss", cls_loss, n_iter)

                loss_temp = 0
                tic = time.time()

        if epoch % args.save_interval == 0:
            save_name = os.path.join(output_dir, f'yolov2_epoch_{epoch}.pth')
            torch.save({
                'model': model.module.state_dict() if args.mGPUs else model.state_dict(),
                'epoch': epoch,
                'lr': lr
            }, save_name)
        scheduler.step()


if __name__ == "__main__":
    start_train()
    config((416, 416))

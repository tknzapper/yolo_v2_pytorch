from voc2007 import *
from torch.utils.data import DataLoader
from loss import *
import torchvision.transforms as transforms
import torch
import math
import albumentations as A
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard.writer import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter('./runs/test')
transform = A.Compose([
    # A.RandomSizedBBoxSafeCrop(cfg.resize, cfg.resize, erosion_rate=0.2, p=1),
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.Resize(cfg.resize, cfg.resize, p=1),
    A.Normalize(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_dataset = VOCDataset(data_root=data_root,
                           transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          collate_fn=detection_collate)

model = Yolo_v2().to(device)
criterion = Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

iters_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    loss_temp = 0
    for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        ouputs = model(inputs)
        box_loss, conf_loss, noobj_loss, cls_loss = criterion(ouputs, targets)
        loss = box_loss.mean() + conf_loss.mean() + noobj_loss.mean() + cls_loss.mean()
        # accuracy = ""

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_temp += loss.item()
        if (i + 1) % 10 == 0:
            loss_temp /= 10
            box_loss_x = box_loss.mean().item()
            conf_loss_x = conf_loss.mean().item()
            noobj_loss_x = noobj_loss.mean().item()
            cls_loss_x = cls_loss.mean().item()

            print("[epoch %2d][step %3d/%3d] loss: %.4f" \
                  % (epoch, i, iters_per_epoch, loss))
            print("\tbox_loss: %.4f, conf_loss: %.4f, noobj_loss: %.4f, cls_loss: %.4f" \
                  % (box_loss_x, conf_loss_x, noobj_loss_x, cls_loss_x))
            print()

            n_iter = epoch * iters_per_epoch + i + 1
            writer.add_scalar("losses/loss", loss_temp, n_iter)
            writer.add_scalar("losses/box_loss", box_loss_x, n_iter)
            writer.add_scalar("losses/conf_loss", conf_loss_x, n_iter)
            writer.add_scalar("losses/noobj_loss", noobj_loss_x, n_iter)
            writer.add_scalar("losses/cls_loss", cls_loss_x, n_iter)

            loss_temp = 0

if __name__ == "__main__":

    for epoch in range(cfg.epochs):
        train(train_loader, model, criterion, optimizer, epoch)

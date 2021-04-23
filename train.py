import os

from voc2007 import *
from torch.utils.data import DataLoader
from loss import *
import torchvision.transforms as transforms
import torch
import math
import time
import albumentations as A
from torch.utils.tensorboard.writer import SummaryWriter

# torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

writer = SummaryWriter()
transform = A.Compose([
    # A.RandomSizedBBoxSafeCrop(cfg.resize, cfg.resize, erosion_rate=0.2, p=1),
    # A.RandomBrightnessContrast(p=0.5),
    # A.HueSaturationValue(p=0.5),
    A.Resize(cfg.resize, cfg.resize, p=1),
    A.Normalize(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

train_root = cfg.data_root + 'Images/Train/'
valid_root = cfg.data_root + 'Images/Validation/'
test_root = cfg.data_root + 'Images/Test/'

root = os.path.join(cfg.data_root, cfg.img_root)

train_dataset = VOCDataset(img_root=root,
                           transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=len(train_dataset),
                          shuffle=True,
                          collate_fn=detection_collate)

val_dataset = VOCDataset(img_root=valid_root,
                         transform=transform)

val_loader = DataLoader(dataset=val_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=detection_collate)

model = Yolo_v2(pretrained=False).to(device)
# PATH = os.path.join(cfg.save_path, 'model.pth')
# model.load_state_dict(torch.load(PATH, map_location=device))
criterion = Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1, verbose=True)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, verbose=True)
iters_per_epoch = math.ceil(len(train_dataset) / cfg.batch_size)

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    loss_temp = 0
    for i, (inputs, targets) in enumerate(train_loader, 1):
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        ouputs = model(inputs)

        box_loss, conf_loss, noobj_loss, cls_loss = criterion(ouputs, targets)
        loss = box_loss + conf_loss + noobj_loss + cls_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_temp += loss.item()
        # if (i + 1) % 10 == 0:
        #     loss_temp /= 10

        print("[epoch %2d][step %3d/%3d] loss: %.4f" \
              % (epoch, i, iters_per_epoch, loss))
        print("\tbox_loss: %.4f, conf_loss: %.4f, noobj_loss: %.4f, cls_loss: %.4f" \
              % (box_loss, conf_loss, noobj_loss, cls_loss))
        print()

        if epoch > 0:
            n_iter = (epoch - 1) * iters_per_epoch + i + 1
            writer.add_scalar("losses/loss", loss_temp, n_iter)
            writer.add_scalar("losses/box_loss", box_loss, n_iter)
            writer.add_scalar("losses/conf_loss", conf_loss, n_iter)
            writer.add_scalar("losses/noobj_loss", noobj_loss, n_iter)
            writer.add_scalar("losses/cls_loss", cls_loss, n_iter)

        loss_temp = 0

def valid(val_loader, model, criterion, epoch):
    model.eval()
    loss_mean = []
    loss_temp = 0
    for i, (inputs, targets) in enumerate(val_loader, 1):
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        ouputs = model(inputs)
        box_loss, conf_loss, noobj_loss, cls_loss = criterion(ouputs, targets)
        loss = box_loss + conf_loss + noobj_loss + cls_loss
        loss_temp += loss.item()
        if (i + 1) % 10 == 0:
            loss_temp /= 10
            loss_mean.append(loss_temp)
            loss_temp = 0
    val_loss = sum(loss_mean, 0.0) / len(loss_mean)
    print("val_loss: %.4f" % (val_loss))
    writer.add_scalar("losses/val_loss", val_loss, epoch)
    return val_loss

best_loss = []
def save_model(model, val_loss):
    PATH = os.path.join(cfg.save_path, 'model.pth')
    best_loss.append(val_loss)
    if min(best_loss) == val_loss:
        torch.save(model.state_dict(), PATH)
        print("Model saved. val_loss: {:.4f}".format(val_loss))
    else:
        print("Skip model save. Best val_loss: {:.4f}".format(min(best_loss)))


if __name__ == "__main__":

    for epoch in range(cfg.epochs):
        train(train_loader, model, criterion, optimizer, epoch)
        # val_loss = valid(val_loader, model, criterion, epoch)
        # save_model(model, val_loss)
        # scheduler.step()
    PATH = os.path.join(cfg.save_path, 'model.pth')
    torch.save(model.state_dict(), PATH)
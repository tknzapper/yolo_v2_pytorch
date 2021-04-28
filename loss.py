from model import *
from utils import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        # self.bce = nn.BCELoss(reduction='sum')
        self.cel = nn.CrossEntropyLoss(reduction='sum')
        self.anchor = torch.FloatTensor(anchor_box)
        self.num_classes = num_classes

    def forward(self, prediction, target):
        bsize, _, h, w = prediction.size()
        out = prediction.permute(0, 2, 3, 1).contiguous().view(bsize, h, w, len(self.anchor), 5 + self.num_classes)
        target.unsqueeze_(3)

        # sorting prediction data
        pred_conf = torch.sigmoid(out[..., 20])
        pred_xy = torch.sigmoid(out[..., 21:23])
        pred_wh = torch.exp(out[..., 23:25]) * self.anchor[..., 0:2].to(device)
        pred_cls = out[..., :20]
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)

        # sorting target data
        gt_conf = target[..., 20]
        gt_xy = target[..., 21:23]
        gt_wh = target[..., 23:25]
        gt_cls = target[..., :20]
        gt_box = torch.cat([gt_xy, gt_wh], dim=-1)

        # get gt index
        idx = torch.nonzero(gt_conf)
        num_obj = len(idx)
        batch = idx[:, 0].T
        grid_y_idx = idx[:, 1].T
        grid_x_idx = idx[:, 2].T

        # get best anchor box index
        iou = box_iou(pred_box, gt_box)
        argmax_idx = torch.argmax(iou, dim=-1, keepdim=True)
        argmax_anchor_idx = argmax_idx[batch, grid_y_idx, grid_x_idx].T

        # for non-object
        inv = 1 - gt_conf
        noobj_idx = torch.nonzero(inv)
        num_nonobj = len(noobj_idx)
        noobj_batch = noobj_idx[:, 0].T
        noobj_grid_y = noobj_idx[:, 1].T
        noobj_grid_x = noobj_idx[:, 2].T
        noobj_anchor_idx = argmax_idx[noobj_batch, noobj_grid_y, noobj_grid_x].T

        # localization loss
        bbox_pred = pred_box[batch, grid_y_idx, grid_x_idx, argmax_anchor_idx]
        bbox_pred.squeeze_(0)
        bbox_gt = gt_box[batch, grid_y_idx, grid_x_idx, 0]
        box_loss = 1 / bsize * lambda_coord * self.mse(bbox_pred, bbox_gt)
        # print(bbox_pred.tolist()[0])

        # object loss
        obj_pred_conf = pred_conf[batch, grid_y_idx, grid_x_idx, argmax_anchor_idx]
        obj_pred_conf.squeeze_(0)
        obj_gt_conf = gt_conf[batch, grid_y_idx, grid_x_idx, 0]
        conf_loss = 1 / bsize * lambda_obj * self.mse(obj_pred_conf, obj_gt_conf)
        # print(obj_pred_conf.tolist()[0])

        # non-object loss
        noobj_pred_conf = pred_conf[noobj_batch, noobj_grid_y, noobj_grid_x, noobj_anchor_idx]
        noobj_pred_conf.squeeze_(0)
        noobj_gt_conf = torch.zeros_like(noobj_pred_conf)
        noobj_loss = 1 / bsize * lambda_noobj * self.mse(noobj_pred_conf, noobj_gt_conf)

        # classification loss
        cls_pred = pred_cls[batch, grid_y_idx, grid_x_idx, argmax_anchor_idx]
        cls_pred.squeeze_(0)
        cls_gt = torch.argmax(gt_cls[batch, grid_y_idx, grid_x_idx, 0], dim=1)
        cls_loss = 1 / bsize * lambda_cls * self.cel(cls_pred, cls_gt)

        return box_loss, conf_loss, noobj_loss, cls_loss


if __name__ == "__main__":
    from voc2007 import *
    from torch.utils.data import DataLoader
    import albumentations as A

    transform = A.Compose([
        # A.RandomSizedBBoxSafeCrop(cfg.resize, cfg.resize, erosion_rate=0.2, p=1),
        # A.RandomBrightnessContrast(p=0.5),
        # A.HueSaturationValue(p=0.5),
        A.Resize(cfg.resize, cfg.resize, p=1),
        A.Normalize(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    # train_root = os.path.join(cfg.data_root, cfg.img_root)
    train_root = os.path.join(cfg.data_root, 'Images/Test/')

    train_dataset = VOCDataset(img_root=train_root,
                               transform=transform)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=len(train_dataset),
                              shuffle=False,
                              collate_fn=detection_collate)

    model = Yolo_v2(pretrained=True).to(device)
    criterion = Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    epochs = 1000
    model.train()
    for epoch in range(epochs):
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            box_loss, conf_loss, noobj_loss, cls_loss = criterion(outputs, labels)
            loss = box_loss + conf_loss + noobj_loss + cls_loss

            # print("[epoch {:2d}] loss: {:.4f}".format(epoch, loss))
            # print("\tbox_loss: {:.4f}, conf_loss: {:.4f}, noobj_loss: {:.4f}, cls_loss: {:.4f}"
            #       .format(box_loss, conf_loss, noobj_loss, cls_loss))
            # print()

            loss.backward()
            optimizer.step()

            # break
        # break

    # PATH = os.path.join(cfg.save_path, 'model.pth')
    # torch.save(model.state_dict(), PATH)
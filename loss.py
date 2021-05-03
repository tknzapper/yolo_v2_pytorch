import config as cfg
from utils import box_iou, generate_anchorbox
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.softmax = nn.Softmax()
        self.cel = nn.CrossEntropyLoss(reduction='sum')
        self.anchor = cfg.anchor_box
        self.num_classes = cfg.num_classes

    def forward(self, prediction, target):
        bsize, _, h, w = prediction.size()
        out = prediction.permute(0, 2, 3, 1).contiguous().view(bsize, h, w, len(self.anchor), 5 + self.num_classes)
        device = out.device
        target.unsqueeze_(3)

        # sorting prediction data
        pred_conf = torch.sigmoid(out[..., 20])
        pred_xy = torch.sigmoid(out[..., 21:23])
        pred_wh = torch.exp(out[..., 23:25])
        pred_cls = out[..., :20]
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
        anchor_box = torch.FloatTensor(self.anchor).to(device)
        pred_box = generate_anchorbox(pred_box, anchor_box)

        # sorting target data
        gt_conf = target[..., 20]
        gt_xy = target[..., 21:23]
        gt_wh = target[..., 23:25]
        gt_cls = target[..., :20]
        gt_box = torch.cat([gt_xy, gt_wh], dim=-1)

        # get gt index
        idx = torch.nonzero(gt_conf)
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
        noobj_batch = noobj_idx[:, 0].T
        noobj_grid_y = noobj_idx[:, 1].T
        noobj_grid_x = noobj_idx[:, 2].T
        noobj_anchor_idx = argmax_idx[noobj_batch, noobj_grid_y, noobj_grid_x].T

        # localization loss
        bbox_pred = pred_box[batch, grid_y_idx, grid_x_idx, argmax_anchor_idx]
        bbox_pred.squeeze_(0)
        bbox_gt = gt_box[batch, grid_y_idx, grid_x_idx, 0]
        box_loss = 1 / bsize * cfg.lambda_coord * self.mse(bbox_pred, bbox_gt)
        # print(bbox_pred.tolist()[0])

        # object loss
        obj_pred_conf = pred_conf[batch, grid_y_idx, grid_x_idx, argmax_anchor_idx]
        obj_pred_conf.squeeze_(0)
        obj_gt_conf = gt_conf[batch, grid_y_idx, grid_x_idx, 0]
        conf_loss = 1 / bsize * cfg.lambda_obj * self.mse(obj_pred_conf, obj_gt_conf)
        # print(obj_pred_conf.tolist()[0])

        # non-object loss
        noobj_pred_conf = pred_conf[noobj_batch, noobj_grid_y, noobj_grid_x, noobj_anchor_idx]
        noobj_pred_conf.squeeze_(0)
        noobj_gt_conf = torch.zeros_like(noobj_pred_conf)
        noobj_loss = 1 / bsize * cfg.lambda_noobj * self.mse(noobj_pred_conf, noobj_gt_conf)

        # classification loss
        cls_pred = pred_cls[batch, grid_y_idx, grid_x_idx, argmax_anchor_idx]
        cls_pred.squeeze_(0)
        cls_gt = torch.argmax(gt_cls[batch, grid_y_idx, grid_x_idx, 0], dim=1)
        cls_loss = 1 / bsize * cfg.lambda_cls * self.cel(cls_pred, cls_gt)

        return box_loss, conf_loss, noobj_loss, cls_loss
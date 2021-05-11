from config import config as cfg
from utils.bbox import box_iou, box_transform_inv, generate_grid
import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.cel = nn.CrossEntropyLoss(reduction='sum')
        self.anchor = cfg.anchor_box
        self.num_classes = cfg.num_classes

    def forward(self, out, gt_data):
        bsize, _, H, W = out.size()
        out = out.permute(0, 2, 3, 1).contiguous().view(-1, len(self.anchor), 5 + self.num_classes)
        device = out.device
        anchor = torch.FloatTensor(self.anchor).to(device)
        grid_xywh = generate_grid(bsize, H, W)
        grid_xywh = out.new(*grid_xywh.size()).copy_(grid_xywh)
        grid_xywh_pred = grid_xywh.view(bsize * H * W, 1, 2).expand(bsize * H * W, len(self.anchor), 2)
        grid_xywh_target = grid_xywh.view(bsize * H * W, 1, 2)

        # prediction data
        delta_xy = torch.sigmoid(out[..., 21:23])
        delta_wh = torch.exp(out[..., 23:25])
        delta_box = torch.cat([delta_xy, delta_wh], dim=-1)
        delta_box[..., 2:4] *= anchor[..., 0:2]
        pred_box = box_transform_inv(grid_xywh_pred, delta_box)
        pred_conf = torch.sigmoid(out[..., 20])
        pred_cls = out[..., :20]

        # target data
        target = self.build_target(gt_data, bsize, H, W)
        target = target.view(-1, 5 + self.num_classes)
        target.unsqueeze_(1)
        gt_box = target[..., 21:25]
        target_box = box_transform_inv(grid_xywh_target, gt_box)
        target_conf = target[..., 20]
        target_cls = target[..., :20]

        # get best anchor box index
        iou = box_iou(pred_box, target_box)
        argmax_idx = torch.argmax(iou, dim=-1, keepdim=True)
        idx = torch.nonzero(target_conf.squeeze()).squeeze()
        argmax_anchor_idx = argmax_idx[idx].squeeze()

        # for non-object
        mask = target_conf.new_zeros(bsize * H * W, len(self.anchor))
        mask[idx, argmax_anchor_idx] = 1
        inv_mask = 1 - mask
        noobj_idx = torch.nonzero(inv_mask).squeeze()

        # localization loss
        bbox_pred = pred_box[idx, argmax_anchor_idx]
        bbox_pred_xy = bbox_pred[..., 0:2]
        bbox_pred_wh = torch.sqrt(bbox_pred[..., 2:4])
        bbox_pred = torch.cat([bbox_pred_xy, bbox_pred_wh], dim=-1)
        bbox_target = target_box[idx, 0]
        bbox_target_xy = bbox_target[..., 0:2]
        bbox_target_wh = torch.sqrt(bbox_target[..., 2:4])
        bbox_target = torch.cat([bbox_target_xy, bbox_target_wh], dim=-1)
        box_loss = 1 / bsize * cfg.lambda_coord * self.mse(bbox_pred, bbox_target)

        # confidence loss
        conf_pred = pred_conf[idx, argmax_anchor_idx]
        conf_target = target_conf[idx, 0]
        conf_loss = 1 / bsize * cfg.lambda_obj * self.mse(conf_pred, conf_target)

        # non-object loss
        noobj_pred = pred_conf[noobj_idx[:, 0], noobj_idx[:, 1]]
        noobj_target = torch.zeros_like(noobj_pred)
        noobj_loss = 1 / bsize * cfg.lambda_noobj * self.mse(noobj_pred, noobj_target)

        # classification loss
        cls_pred = pred_cls[idx, argmax_anchor_idx]
        cls_target = torch.argmax(target_cls[idx, 0], dim=-1)
        cls_loss = 1 / bsize * cfg.lambda_cls * self.cel(cls_pred, cls_target)

        return box_loss, conf_loss, noobj_loss, cls_loss

    def build_target(self, gt_data, bsize, H, W):
        gt_boxes = gt_data[0]
        gt_classes = gt_data[1]
        num_box = gt_data[2]

        target = gt_boxes.new_zeros((bsize, H, W, (self.num_classes + 5)))

        for b in range(bsize):
            num_obj = num_box[b].item()
            gt_bbox = gt_boxes[b][:num_obj, :]
            gt_cls = gt_classes[b][:num_obj]
            for objs in range(num_obj):
                bbox = gt_bbox[objs]
                cls = gt_cls[objs].long()
                objectness = 1
                scale_factor = 1 / H
                grid_x_idx = int(bbox[0] // scale_factor)
                grid_y_idx = int(bbox[1] // scale_factor)
                x_offset = bbox[0] / scale_factor - grid_x_idx
                y_offset = bbox[1] / scale_factor - grid_y_idx
                w = bbox[2] / scale_factor
                h = bbox[3] / scale_factor
                target[b, grid_y_idx, grid_x_idx, self.num_classes:self.num_classes + 5] = \
                    torch.FloatTensor([objectness, x_offset, y_offset, w, h])
                target[b, grid_y_idx, grid_x_idx, cls] = 1

        return target
import torch
from config import config as cfg


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, conf_pred=None, conf_cls=None, cls=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.conf_pred = conf_pred
        self.conf_cls = conf_cls
        self.cls = cls
        self.cls_name = cfg.classes[self.cls]

        self.label = -1
        self.score = -1


def filter_boxes(pred_box, pred_conf, pred_cls, confidence_threshold):
    box_pred = pred_box.view(-1, 4)
    conf_pred = pred_conf.view(-1, 1)
    cls_pred = pred_cls.view(-1, 20)

    cls_max_conf, cls_max_id = torch.max(cls_pred, dim=1, keepdim=True)
    cls_conf = conf_pred * cls_max_conf

    pos_idxs = (cls_conf > confidence_threshold).view(-1)

    filtered_boxes = box_pred[pos_idxs, :]
    filtered_conf = conf_pred[pos_idxs, :]
    filtered_cls_max_conf = cls_max_conf[pos_idxs, :]
    filtered_cls_max_id = cls_max_id[pos_idxs, :]

    return filtered_boxes, filtered_conf, filtered_cls_max_conf, filtered_cls_max_id


def box_iou(box1, box2):
    xmin = torch.max(box1[..., 0], box2[..., 0])
    ymin = torch.max(box1[..., 1], box2[..., 1])
    xmax = torch.min(box1[..., 2], box2[..., 2])
    ymax = torch.min(box1[..., 3], box2[..., 3])

    iw = torch.max(xmax - xmin, box1.new_zeros(1))
    ih = torch.max(ymax - ymin, box1.new_zeros(1))

    inter_area = iw * ih
    inter_area.squeeze_()

    box1_area = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    box2_area = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


def nms(boxes, scores, threshold):
    score_sort_idx = torch.sort(scores, dim=0, descending=True)[1]
    keep = []
    while score_sort_idx.numel() > 0:
        i = score_sort_idx[0]
        keep.append(i)
        if score_sort_idx.numel() == 1:
            break
        cur_box = boxes[score_sort_idx[0], :]
        res_box = boxes[score_sort_idx[1:], :]

        iou = box_iou(cur_box, res_box).view(-1)
        idxs = torch.nonzero(iou < threshold).squeeze()
        score_sort_idx = score_sort_idx[idxs + 1].view(-1)

    return torch.LongTensor(keep)


def eval(pred_box, pred_conf, pred_cls, conf_threshold, nms_threshold):
    box_pred, conf_pred, max_conf, max_id = filter_boxes(pred_box, pred_conf, pred_cls, conf_threshold)

    if box_pred.size(0) == 0:
        return []

    detections = []

    max_id = max_id.view(-1)
    for cls in range(cfg.num_classes):
        cls_mask = max_id == cls
        idxs = torch.nonzero(cls_mask).squeeze()

        if idxs.numel() == 0:
            continue

        box_pred_cls = box_pred[idxs, :].view(-1, 4)
        conf_pred_cls = conf_pred[idxs, :].view(-1, 1)
        cls_conf = max_conf[idxs].view(-1, 1)
        cls_id = max_id[idxs].view(-1, 1)

        nms_keep = nms(box_pred_cls, conf_pred_cls.view(-1), nms_threshold)

        pred_box_keep = box_pred_cls[nms_keep, :]
        pred_conf_keep = conf_pred_cls[nms_keep, :]
        cls_conf_keep = cls_conf.view(-1, 1)[nms_keep, :]
        cls_id_keep = cls_id.view(-1, 1)[nms_keep, :]

        res = torch.cat([pred_box_keep, pred_conf_keep, cls_conf_keep, cls_id_keep], dim=-1)
        detections.append(res)

    return torch.cat(detections, dim=0)
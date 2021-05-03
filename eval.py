import torch
import utils
import config as cfg


def filter_boxes(pred_box, pred_conf, pred_cls, confidence_threshold=0.6):
    box_pred = pred_box.view(-1, 4)
    conf_pred = pred_conf.view(-1, 1)
    cls_pred = pred_cls.view(-1, 20)

    cls_max_conf, cls_max_id = torch.max(cls_pred, dim=1, keepdim=True)
    cls_conf = conf_pred * cls_max_conf

    pos_inds = (cls_conf > confidence_threshold).view(-1)

    filtered_boxes = box_pred[pos_inds, :]
    filtered_conf = conf_pred[pos_inds, :]
    filtered_cls_max_conf = cls_max_conf[pos_inds, :]
    filtered_cls_max_id = cls_max_id[pos_inds, :]

    return filtered_boxes, filtered_conf, filtered_cls_max_conf, filtered_cls_max_id.float()


def nms(boxes, scores, threshold):
    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:
        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1], :].view(-1, 4)

        ious = utils.box_iou(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)


def generate_prediction_boxes(box):
    box_ = box.clone()
    box_.squeeze_(0)
    feature_size = int(cfg.resize[0] / cfg.scale_size)
    scale_factor = 1 / feature_size

    for grid_y in range(box_.size(0)):
        for grid_x in range(box_.size(1)):
            box_[grid_y, grid_x, :, 0] = (box[0, grid_y, grid_x, :, 0] + grid_x) * scale_factor     # x
            box_[grid_y, grid_x, :, 1] = (box[0, grid_y, grid_x, :, 1] + grid_y) * scale_factor     # y
            box_[grid_y, grid_x, :, 2] = box[0, grid_y, grid_x, :, 2] * scale_factor                # w
            box_[grid_y, grid_x, :, 3] = box[0, grid_y, grid_x, :, 3] * scale_factor                # h

    box_ = box_.view(-1, 4)
    mask_small = box_[:, 2] * box_[:, 3] > 1 / feature_size
    box_ = utils.xywh2xxyy(box_)
    mask_big = torch.cat([box_[:, 0:2] >= 0, box_[:, 2:4] <= 1], dim=1).int()
    mask_big = mask_big[:, 0] * mask_big[:, 1] * mask_big[:, 2] * mask_big[:, 3]
    mask_size = torch.logical_and(mask_small.int(), mask_big)

    idx = torch.nonzero(mask_size).squeeze_()
    box_ = torch.clamp(box_, min=0, max=1)

    return box_, idx


def eval(pred_box, pred_conf, pred_cls, conf_threshold, nms_threshold):
    pred_box, idx = generate_prediction_boxes(pred_box)
    pred_conf = pred_conf.view(-1, 1)
    pred_cls = pred_cls.view(-1, 20)

    pred_box = pred_box[idx]
    pred_conf = pred_conf[idx]
    pred_cls = pred_cls[idx]

    box_pred, conf_pred, max_conf, max_id = filter_boxes(pred_box, pred_conf, pred_cls, conf_threshold)

    if box_pred.size(0) == 0:
        return []

    detections = []

    max_id = max_id.view(-1)
    for cls in range(cfg.num_classes):
        cls_mask = max_id == cls
        inds = torch.nonzero(cls_mask).squeeze()

        if inds.numel() == 0:
            continue

        boxes_pred_class = box_pred[inds, :].view(-1, 4)
        conf_pred_class = conf_pred[inds, :].view(-1, 1)
        cls_max_conf_class = max_conf[inds].view(-1, 1)
        classes_class = max_id[inds].view(-1, 1)

        nms_keep = nms(boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

        boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
        conf_pred_class_keep = conf_pred_class[nms_keep, :]
        cls_max_conf_class_keep = cls_max_conf_class.view(-1, 1)[nms_keep, :]
        classes_class_keep = classes_class.view(-1, 1)[nms_keep, :]

        seq = [boxes_pred_class_keep, conf_pred_class_keep, cls_max_conf_class_keep, classes_class_keep.float()]

        detections_cls = torch.cat(seq, dim=-1)
        detections.append(detections_cls)

    return torch.cat(detections, dim=0)
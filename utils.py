import torch


def xxyy2xywh(box):
    """
    box = [xmin, ymin, xmax, ymax]
    :param box: xmin, ymin, xmax, ymax
    :return: cx, cy, w, h
    """
    box_ = box.clone()
    box[..., 0] = (box_[..., 2] + box_[..., 0]) / 2
    box[..., 1] = (box_[..., 3] + box_[..., 1]) / 2
    box[..., 2] = box_[..., 2] - box_[..., 0]
    box[..., 3] = box_[..., 3] - box_[..., 1]

    return box


def xywh2xxyy(box):
    """
    box = [cx, cy, w, h]
    :param box: cx, cy, w, h
    :return: xmin, ymin, xmax, ymax
    """
    box_ = box.clone()
    box[..., 0] = box_[..., 0] - (box_[..., 2] / 2)
    box[..., 1] = box_[..., 1] - (box_[..., 3] / 2)
    box[..., 2] = box_[..., 0] + (box_[..., 2] / 2)
    box[..., 3] = box_[..., 1] + (box_[..., 3] / 2)

    return box


def box_iou(box1, box2):
    """
    box size: [batch, grid, anchor, coordinate]
    :param box1: [cx1, cy1, w1, h1]
    :param box2: [cx2, cy2, w2, h2]
    :return:
    """
    box1_ = box1.clone()
    box2_ = box2.clone()
    xywh2xxyy(box1_)
    xywh2xxyy(box2_)

    xmin = torch.maximum(box1_[..., 0], box2_[..., 0])
    ymin = torch.maximum(box1_[..., 1], box2_[..., 1])
    xmax = torch.minimum(box1_[..., 2], box2_[..., 2])
    ymax = torch.minimum(box1_[..., 3], box2_[..., 3])

    W = xmax - xmin
    H = ymax - ymin
    W = torch.where(W > 0, W, torch.zeros_like(W))
    H = torch.where(H > 0, H, torch.zeros_like(H))

    area_intersect = W * H
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    area_union = area1 + area2 - area_intersect

    return area_intersect / (area_union + 1e-4)


def generate_anchorbox(box, anchor):
    """
    box size: [batch, grid, anchor, coordinate]
    :param box: [x, y, w, h]
    :return:
    """

    box[..., 2:4] *= anchor[..., 0:2]

    return box
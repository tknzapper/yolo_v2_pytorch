import torch
from config import *



def xxyy2xywh(box):
    """
    box = [xmin, ymin, xmax, ymax]
    :param box: xmin, ymin, xmax, ymax
    :return: cx, cy, w, h
    """

    cx = (box[2] + box[0]) / 2 - 1
    cy = (box[3] + box[1]) / 2 - 1
    w = box[2] - box[0]
    h = box[3] - box[1]

    return [cx, cy, w, h]


def xywh2xxyy(box):
    """
    box = [cx, cy, w, h]
    :param box: cx, cy, w, h
    :return: xmin, ymin, xmax, ymax
    """

    xmin = box[0] - (box[2] / 2)
    ymin = box[1] - (box[3] / 2)
    xmax = box[0] + (box[2] / 2)
    ymax = box[1] + (box[3] / 2)

    return [xmin, ymin, xmax, ymax]


def normalize(box, w, h):
    box[0] /= w
    box[1] /= h
    box[2] /= w
    box[3] /= h

    return box


def box_iou(box1, box2):
    """
    box size: [batch, grid, anchor, coordinate]
    :param box1: [cx1, cy1, w1, h1]
    :param box2: [cx2, cy2, w2, h2]
    :return:
    """

    x1max = box1[..., 0] + box1[..., 2] / 2
    x1min = box1[..., 0] - box1[..., 2] / 2
    y1max = box1[..., 1] + box1[..., 3] / 2
    y1min = box1[..., 1] - box1[..., 3] / 2

    x2max = box2[..., 0] + box2[..., 2] / 2
    x2min = box2[..., 0] - box2[..., 2] / 2
    y2max = box2[..., 1] + box2[..., 3] / 2
    y2min = box2[..., 1] - box2[..., 3] / 2

    xmax = torch.minimum(x1max, x2max)
    xmin = torch.maximum(x1min, x2min)
    ymax = torch.minimum(y1max, y2max)
    ymin = torch.maximum(y1min, y2min)

    area_intersect = (xmax - xmin) * (ymax - ymin)
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    area_union = area1 + area2 - area_intersect

    return area_intersect / (area_union + 1e-6)


def generate_anchorbox(box, device="cuda"):
    """
    box size: [batch, grid, anchor, coordinate]
    :param box: [x, y, w, h]
    :return:
    """

    anchor = torch.FloatTensor(anchor_box).to(device)

    box[..., 2:4] *= anchor[..., 0:2]

    return box
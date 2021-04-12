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

    xmin = box[0] - (box[2] / 2) + 1
    ymin = box[1] - (box[3] / 2) + 1
    xmax = box[0] + (box[2] / 2) + 1
    ymax = box[1] + (box[3] / 2) + 1

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

    x1max = box1[:, :, :, 0] + box1[:, :, :, 2] / 2
    x1min = box1[:, :, :, 0] - box1[:, :, :, 2] / 2
    y1max = box1[:, :, :, 1] + box1[:, :, :, 3] / 2
    y1min = box1[:, :, :, 1] - box1[:, :, :, 3] / 2

    x2max = box2[:, :, :, 0] + box2[:, :, :, 2] / 2
    x2min = box2[:, :, :, 0] - box2[:, :, :, 2] / 2
    y2max = box2[:, :, :, 1] + box2[:, :, :, 3] / 2
    y2min = box2[:, :, :, 1] - box2[:, :, :, 3] / 2

    xmax = torch.max(x1max, x2max)
    xmin = torch.min(x1min, x2min)
    ymax = torch.max(y1max, y2max)
    ymin = torch.min(y1min, y2min)

    area_intersect = (xmax - xmin) * (ymax - ymin)
    area1 = box1[:, :, :, 2] * box1[:, :, :, 3]
    area2 = box2[:, :, :, 2] * box2[:, :, :, 3]
    area_union = area1 + area2 - area_intersect

    iou = area_intersect / area_union
    iou = torch.where(iou >= 0, iou, iou.new_zeros((iou.size(0), iou.size(1), iou.size(2))))
    iou = torch.where(iou <= 1, iou, iou.new_zeros((iou.size(0), iou.size(1), iou.size(2))))

    return iou


def delta_box(box1, box2):
    """
    box size: [batch, grid, anchor, coordinate]
    :param box1: [cx1, cy1, w1, h1]
    :param box2: [cx2, cy2, w2, h2]
    :return: delta(box1, box2)
    """

    bsize, gsize, asize, _ = box1.size()

    tx = box2[:, :, :, 0] - box1[:, :, :, 0]
    ty = box2[:, :, :, 1] - box1[:, :, :, 1]
    tw = box2[:, :, :, 2] / (box1[:, :, :, 2] + 1e-6)
    th = box2[:, :, :, 3] / (box1[:, :, :, 3] + 1e-6)

    tx = tx.view(bsize, gsize, asize, -1)
    ty = ty.view(bsize, gsize, asize, -1)
    tw = tw.view(bsize, gsize, asize, -1)
    th = th.view(bsize, gsize, asize, -1)

    return torch.cat([tx, ty, tw, th], dim=-1)


def generate_anchorbox(box, device="cuda"):
    """
    box size: [batch, grid, anchor, coordinate]
    :param box: [x, y, w, h]
    :return:
    """

    anchor = torch.FloatTensor(anchor_box).to(device)

    box[:, :, :, 2:4] *= anchor[:, 0:2]

    return box

import torch
from config import *



def xxyy2xywh(box):
    """
    box = [xmin, ymin, xmax, ymax]
    :param box: xmin, ymin, xmax, ymax
    :return: cx, cy, w, h
    """

    cx = (box[:, 2] + box[:, 0]) / 2
    cy = (box[:, 3] + box[:, 1]) / 2
    w = box[:, 2] - box[:, 0]
    h = box[:, 3] - box[:, 1]


    cx = cx.view(-1, 1)
    cy = cy.view(-1, 1)
    w = w.view(-1, 1)
    h = h.view(-1, 1)

    return torch.cat([cx, cy, w, h], dim=1)


def xywh2xxyy(box):
    """
    box = [cx, cy, w, h]
    :param box: cx, cy, w, h
    :return: xmin, ymin, xmax, ymax
    """

    xmin = box[:, 0] - (box[:, 2] / 2)
    ymin = box[:, 1] - (box[:, 3] / 2)
    xmax = box[:, 2] + (box[:, 2] / 2)
    ymax = box[:, 1] + (box[:, 3] / 2)

    xmin = xmin.view(-1, 1)
    ymin = ymin.view(-1, 1)
    xmax = xmax.view(-1, 1)
    ymax = ymax.view(-1, 1)

    return torch.cat([xmin, ymin, xmax, ymax], dim=1)


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

    return area_intersect / (area_union + 1e-6)
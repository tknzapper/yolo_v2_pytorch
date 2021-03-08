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


    cx = torch.from_numpy(cx).view(-1, 1)
    cy = torch.from_numpy(cy).view(-1, 1)
    w = torch.from_numpy(w).view(-1, 1)
    h = torch.from_numpy(h).view(-1, 1)

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
    N = box1.size(0)
    K = box2.size(0)

    xi1 = torch.max(box1[:, 0].view(N, 1), box2[:, 0].view(1, K))
    yi1 = torch.max(box1[:, 1].view(N, 1), box2[:, 1].view(1, K))
    xi2 = torch.min(box1[:, 2].view(N, 1), box2[:, 2].view(1, K))
    yi2 = torch.min(box1[:, 3].view(N, 1), box2[:, 3].view(1, K))

    iw = torch.max(xi2 - xi1, box1.new(1).fill_(0))
    ih = torch.max(yi2 - yi1, box1.new(1).fill_(0))

    inter = iw * ih

    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    box1_area = box1_area.view(N, 1)
    box2_area = box2_area.view(1, K)

    union_area = box1_area + box2_area - inter

    return inter / union_area
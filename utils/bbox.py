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
    box1_ = xywh2xxyy(box1_)
    box2_ = xywh2xxyy(box2_)

    K = box1_.size(0)
    A = box1_.size(1)

    xmin = torch.max(box1_[..., 0].view(K, A, 1), box2_[..., 0].view(K, 1, 1).expand(K, A, 1))
    ymin = torch.max(box1_[..., 1].view(K, A, 1), box2_[..., 1].view(K, 1, 1).expand(K, A, 1))
    xmax = torch.min(box1_[..., 2].view(K, A, 1), box2_[..., 2].view(K, 1, 1).expand(K, A, 1))
    ymax = torch.min(box1_[..., 3].view(K, A, 1), box2_[..., 3].view(K, 1, 1).expand(K, A, 1))

    iw = torch.max(xmax - xmin, box1_.new_zeros(1))
    ih = torch.max(ymax - ymin, box1_.new_zeros(1))

    inter_area = iw * ih
    inter_area.squeeze_()

    box1_area = (box1_[..., 2] - box1_[..., 0]) * (box1_[..., 3] - box1_[..., 1])
    box2_area = (box2_[..., 2] - box2_[..., 0]) * (box2_[..., 3] - box2_[..., 1])

    box1_area = box1_area.view(K, A)
    box2_area = box2_area.view(K, 1).expand(K, A)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area

    return iou


def box_transform_inv(grid, delta):
    """

    :param box:
    :param delta:
    :return:
    """
    pred_box = delta.clone()
    pred_box[..., 0:2] += grid[..., 0:2]

    return pred_box


def generate_grid(bsize, H, W):
    """
    generates all grid
    :param H: feature size
    :param W: feature size
    :return: [N, grid_x, grid_y, anchor_w, anchor_h], N: H * W * len(anchor)
    """

    K = H * W

    shift_x, shift_y = torch.meshgrid([torch.arange(0, W), torch.arange(0, H)])

    shift_x = shift_x.T.contiguous()
    shift_y = shift_y.T.contiguous()

    grid_x = shift_x.float()
    grid_y = shift_y.float()

    grid = torch.cat([grid_x.view(-1, 1), grid_y.view(-1, 1)], dim=-1)
    grid = grid.view(K, 2).expand(bsize, K, 2)

    return grid
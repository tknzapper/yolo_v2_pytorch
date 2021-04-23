import torch
from torchvision.ops import nms
import config as cfg
import os
import albumentations as A
import voc2007 as voc
import numpy as np
import utils
import visualize as vis
from model import Yolo_v2

device = torch.device("cpu")

PATH = os.path.join(cfg.save_path, 'model.pth')

model = Yolo_v2(pretrained=True)
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()

trans = A.Compose([
    A.Resize(cfg.resize, cfg.resize, p=1),
    A.Normalize(),
])
denorm = A.Compose([
    A.Normalize(mean=(-0.485 / 0.229 * 255, -0.456 / 0.224 * 255, -0.406 / 0.225 * 255),
                std=(1 / 0.229, 1 / 0.224, 1 / 0.225), max_pixel_value=1 / 255.)
])

test_root = cfg.data_root + 'Images/Test/'
root = os.path.join(cfg.data_root, cfg.img_root)
test_dataset = voc.VOCDataset(img_root=root, transform=trans)

for i in range(len(test_dataset)):
    image = test_dataset[i][0]
    trans_img = denorm(image=image)
    im = trans_img['image'].astype(np.uint8)

    img = torch.from_numpy(test_dataset[i][0])
    img = img.permute(2, 0, 1).contiguous()
    img = torch.unsqueeze(img, 0)
    img = img.to(device)

    out = model(img).detach().cpu()
    bsize, _, H, W = out.size()
    out = out.permute(0, 2, 3, 1).contiguous().view(bsize, H, W, 5, 25)

    pred_conf = torch.sigmoid(out[..., 20])
    pred_xy = torch.sigmoid(out[..., 21:23])
    pred_wh = torch.exp(out[..., 23:25])
    pred_cls = torch.nn.Softmax(dim=-1)(out[..., :20])
    pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
    scale_factor = 1 / cfg.feature_size
    idx = torch.where(pred_conf > 0.9)
    num_obj = idx[0].size(0)
    # print(num_obj)
    batch = idx[0].T
    grid_y_idx = idx[1].T
    grid_x_idx = idx[2].T
    anchor_idx = idx[3].T
    utils.generate_anchorbox(pred_box, device)

    bbox = pred_box[batch, grid_y_idx, grid_x_idx, anchor_idx]
    score = pred_conf[batch, grid_y_idx, grid_x_idx, anchor_idx]
    classes = torch.argmax(pred_cls[batch, grid_y_idx, grid_x_idx, anchor_idx], dim=-1)
    # print(pred_box[0, 7, 6])
    # print(pred_conf[0, 7, 6])
    # print(torch.argmax(pred_cls[0, 7, 6], dim=-1))
    cx = (bbox[:, 0] + grid_x_idx) * scale_factor
    cy = (bbox[:, 1] + grid_y_idx) * scale_factor
    w = bbox[:, 2] * scale_factor
    h = bbox[:, 3] * scale_factor

    xmin = torch.zeros_like(cx)
    ymin = torch.zeros_like(cy)
    xmax = torch.ones_like(cx)
    ymax = torch.ones_like(cy)
    for i in range(num_obj):
        xmin[i] = cx[i] - (w[i] / 2) if cx[i] - (w[i] / 2) > 0 else 0
        ymin[i] = cy[i] - (h[i] / 2) if cy[i] - (h[i] / 2) > 0 else 0
        xmax[i] = cx[i] + (w[i] / 2) if cx[i] + (w[i] / 2) < 1 else 1
        ymax[i] = cy[i] + (h[i] / 2) if cy[i] + (h[i] / 2) < 1 else 1

    bbox = torch.cat([xmin.unsqueeze_(1), ymin.unsqueeze_(1), xmax.unsqueeze_(1), ymax.unsqueeze_(1)], dim=1)
    idx_nms = nms(bbox, score, iou_threshold=0.3)

    vis.visualize(im, bbox[idx_nms], classes[idx_nms])

    break
import torch
import config as cfg
import os
import albumentations as A
import voc2007 as voc
import numpy as np
import utils
import visualize as vis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = os.path.join(cfg.save_path, 'model.pth')

model = torch.load(PATH)
model = model.to(device)
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
    utils.generate_anchorbox(pred_box, device='cpu')

    scale_factor = 1 / cfg.feature_size
    idx = torch.where(pred_conf > 0.9)
    batch = idx[0].T
    grid_y_idx = idx[1].T
    grid_x_idx = idx[2].T
    anchor_idx = idx[3].T
    objects = pred_box[batch, grid_y_idx, grid_x_idx, anchor_idx]

    boxes = []
    for i in range(objects.size(0)):
        cx = (objects[i][0].item() + grid_x_idx[i].item()) * scale_factor
        cy = (objects[i][1].item() + grid_y_idx[i].item()) * scale_factor
        w = objects[i][2].item()
        h = objects[i][3].item()
        box = [cx, cy, w, h]
        # print(box)
        boxes.append(box)

    cls = pred_cls[batch, grid_y_idx, grid_x_idx, anchor_idx]
    _, cls = torch.max(cls, dim=-1)

    vis.visualize(im, boxes, cls)

    break
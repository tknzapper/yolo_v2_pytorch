import torch
import config as cfg
import os
import albumentations as A
import voc2007 as voc
import numpy as np
import utils
import visualize as vis

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATH = os.path.join(cfg.save_path, os.listdir(cfg.save_path)[0])

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
test_dataset = voc.VOCDataset(img_root=test_root, transform=trans)

image = test_dataset[0][0]
trans_img = denorm(image=image)
im = trans_img['image'].astype(np.uint8)

img = torch.from_numpy(test_dataset[0][0])
img = img.permute(2, 0, 1).contiguous()
img = torch.unsqueeze(img, 0)
img = img.to(device)

out = model(img).detach().cpu()
bsize, _, h, w = out.size()
out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * 5, 25)

pred_conf = torch.sigmoid(out[:, :, 20:21])
pred_xy = torch.sigmoid(out[:, :, 21:23])
pred_wh = torch.exp(out[:, :, 23:25])
pred_cls = torch.nn.Softmax(dim=-1)(out[:, :, :20])
pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
pred_box = pred_box.view(bsize, h * w, 5, -1)
anc_box = utils.generate_anchorbox(pred_box, device='cpu')
anc_box = anc_box.view(bsize, h * w * 5, -1)

confIdx = torch.where(torch.sigmoid(out[:, :, 20]) > 0.2)

box = anc_box[confIdx[0], confIdx[1], :]
box = box.tolist()

cls = pred_cls[confIdx[0], confIdx[1], :]
_, cls = torch.max(cls, dim=-1)

vis.visualize(im, box, cls)
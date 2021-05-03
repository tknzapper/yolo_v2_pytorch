import argparse
import os
import torch
from torch.utils.data import DataLoader
import config as cfg
import utils
import albumentations as A
from voc2007 import VOCDataset
from visualize import visualize
from model import Yolo_v2
from eval import eval
import time
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v2 eval')
    parser.add_argument('--cuda', dest='use_cuda',
                        default=True, type=bool)
    parser.add_argument('--mGPUs', dest='mGPUs',
                        default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset',
                        default='voc07test', type=str)
    parser.add_argument('--model', dest='model',
                        default='default', type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of workers to load training data',
                        default=1, type=int)
    parser.add_argument('--conf', dest='conf_thresh',
                        default=0.8, type=float)
    parser.add_argument('--nms', dest='nms_thresh',
                        default=0.4, type=float)

    args = parser.parse_args()
    return args


def demo():
    args = parse_args()
    print(f'Called with args: {args}')

    args.weights_file = os.path.join('weights', 'pretrained', 'darknet19.pth')
    args.data_root = os.path.join('data/VOCdevkit/VOC2007/JPEGImages')

    if args.use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    if args.dataset == 'voc07test':
        args.db_name = 'test'
        args.db_year = '2007'

    else:
        raise NotImplementedError

    if args.model == 'default':
        PATH = os.path.join(cfg.output_dir, 'yolov2_voc0712trainval_E160.pth')

    else:
        PATH = os.path.join(cfg.output_dir, args.model)

    transform = A.Compose([
        A.Resize(416, 416, p=1),
        A.Normalize(),
    ])

    test_dataset = VOCDataset(image_set=args.db_name,  year=args.db_year, transform=transform, train=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1,
                             shuffle=False, num_workers=args.num_workers)

    model = Yolo_v2(weights_file=args.weights_file).to(device)
    checkpoint = torch.load(PATH, map_location=device)

    if args.mGPUs:
        model.module.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint['model'])

    model.eval()

    with torch.no_grad():
        for i, (img, image_file) in enumerate(test_loader):
            image = Image.open(os.path.join(args.data_root, image_file[0])).convert('RGB')
            img = img.permute(0, 3, 1, 2).contiguous()
            img = img.to(device)

            tic = time.time()

            out = model(img)
            bsize, _, H, W = out.size()
            out = out.permute(0, 2, 3, 1).contiguous().view(bsize, H, W, len(cfg.anchor_box), 25)

            pred_conf = torch.sigmoid(out[..., 20])
            pred_xy = torch.sigmoid(out[..., 21:23])
            pred_wh = torch.exp(out[..., 23:25])
            pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
            anchor_box = torch.FloatTensor(cfg.anchor_box).to(device)
            pred_box = utils.generate_anchorbox(pred_box, anchor_box)
            pred_cls = torch.nn.Softmax(dim=-1)(out[..., :20])

            detection = eval(pred_box, pred_conf, pred_cls, args.conf_thresh, args.nms_thresh)
            toc = time.time()
            time_cost = toc - tic
            print(f'[{i+1:2d}/{len(test_dataset):2d}]\t\t\t{time_cost:.4f} sec{int(1/time_cost):>10} fps')
            # print(res.detach().cpu())
            if detection == []:
                print("No object")
                continue
            bbox = detection[:, :4].detach().cpu()
            cls = detection[:, 6].detach().cpu()
            visualize(image, bbox, cls)
            # break


if __name__ == "__main__":
    demo()
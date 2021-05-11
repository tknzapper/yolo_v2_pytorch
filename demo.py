import os
import time
import argparse
import torch
from PIL import Image
import albumentations as A
from voc import VOCDataset
from config import config as cfg
from torch.utils.data import DataLoader
from utils.bbox import xywh2xxyy, box_transform_inv, generate_grid
from utils.eval import eval, BoundBox
from utils.visualize import visualize
from model import Yolo_v2


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
                        default=0.1, type=float)
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

    elif args.dataset == 'voc07train':
        args.db_name = 'train'
        args.db_year = '2007'

    else:
        raise NotImplementedError

    if args.model == 'default':
        PATH = os.path.join(cfg.output_dir, 'yolov2_E160.pth')

    else:
        PATH = os.path.join(cfg.output_dir, args.model)

    transform = A.Compose([
        A.Resize(cfg.test_resize[0], cfg.test_resize[1], p=1),
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
            out = out.permute(0, 2, 3, 1).contiguous().view(-1, len(cfg.anchor_box), 5 + cfg.num_classes)
            device = out.device
            anchor = torch.FloatTensor(cfg.anchor_box).to(device)
            grid_xywh = generate_grid(bsize, H, W)
            grid_xywh = out.new(*grid_xywh.size()).copy_(grid_xywh)
            grid_xywh_pred = grid_xywh.view(bsize * H * W, 1, 2).expand(bsize * H * W, len(cfg.anchor_box), 2)

            # prediction data
            delta_xy = torch.sigmoid(out[..., 21:23])
            delta_wh = torch.exp(out[..., 23:25])
            delta_box = torch.cat([delta_xy, delta_wh], dim=-1)
            delta_box[..., 2:4] *= anchor[..., 0:2]
            pred_box = box_transform_inv(grid_xywh_pred, delta_box)
            pred_box /= H
            pred_box = xywh2xxyy(pred_box)
            pred_conf = torch.sigmoid(out[..., 20])
            pred_cls = torch.nn.Softmax(dim=-1)(out[..., :20])

            detection = eval(pred_box, pred_conf, pred_cls, args.conf_thresh, args.nms_thresh)
            toc = time.time()
            time_cost = toc - tic

            print(f'[{i+1:2d}/{len(test_dataset):2d}]\t\t\t{time_cost:.4f} sec{int(1/time_cost):>10} fps')

            if detection == []:
                print("No object")
                continue

            bbox_pred = detection[:, :4].detach().cpu()
            conf_pred = detection[:, 4].detach().cpu()
            conf_cls = detection[:, 5].detach().cpu()
            cls_pred = detection[:, 6].detach().cpu()

            bbox = []
            print('{0:_^24} {1:_^8} {2:_^8} {3:_<10}'.format('BBOX', 'PRED', 'CONF', 'CLASS'))
            # with open(f'./data/output/{image_file[0].split(".")[0]}.txt', 'w') as f:
            for objs in range(len(bbox_pred)):
                xmin = int(bbox_pred.tolist()[objs][0] * image.width)
                ymin = int(bbox_pred.tolist()[objs][1] * image.height)
                xmax = int(bbox_pred.tolist()[objs][2] * image.width)
                ymax = int(bbox_pred.tolist()[objs][3] * image.height)
                score_pred = conf_pred.tolist()[objs]
                score_cls = conf_cls.tolist()[objs]
                cls = int(cls_pred.tolist()[objs])
                # f.write(f"{cls} {score_pred} {xmin} {ymin} {xmax} {ymax}\n")
                box = BoundBox(xmin, ymin, xmax, ymax, score_pred, score_cls, cls)
                bbox.append(box)
                print(f'[  {xmin:3d}  {ymin:3d}  {xmax:3d}  {ymax:3d}  ]', end=' ')
                print(f' {score_pred:.4f} ', end=' ')
                print(f' {score_cls:.4f} ', end=' ')
                print(f'{box.cls_name:<10}')
            print()
            visualize(image, image_file[0], bbox)
            # break


if __name__ == "__main__":
    demo()
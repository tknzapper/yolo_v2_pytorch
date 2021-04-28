import argparse
import torch
import config as cfg
import os
import albumentations as A
from voc2007 import VOCDataset
import utils
from visualize import visualize
from model import Yolo_v2


def parse_args():
    parser = argparse.ArgumentParser(description='Yolo v2 eval')
    parser.add_argument('--cuda', dest='use_cuda',
                        default=False, type=bool)
    parser.add_argument('--dataset', dest='dataset',
                        default='test', type=str)
    parser.add_argument('--model', dest='model',
                        default='default', type=str)

    args = parser.parse_args()
    return args

def filter_boxes(pred_box, pred_conf, pred_cls, confidence_threshold=0.6):
    box_pred = pred_box.view(-1, 4)
    conf_pred = pred_conf.view(-1, 1)
    cls_pred = pred_cls.view(-1, 20)

    cls_max_conf, cls_max_id = torch.max(cls_pred, dim=1, keepdim=True)
    cls_conf = conf_pred * cls_max_conf

    pos_inds = (cls_conf > confidence_threshold).view(-1)

    filtered_boxes = box_pred[pos_inds, :]
    filtered_conf = conf_pred[pos_inds, :]
    filtered_cls_max_conf = cls_max_conf[pos_inds, :]
    filtered_cls_max_id = cls_max_id[pos_inds, :]

    return filtered_boxes, filtered_conf, filtered_cls_max_conf, filtered_cls_max_id.float()

def yolo_nms(boxes, scores, threshold):
    score_sort_index = torch.sort(scores, dim=0, descending=True)[1]

    keep = []

    while score_sort_index.numel() > 0:
        i = score_sort_index[0]
        keep.append(i)

        if score_sort_index.numel() == 1:
            break

        cur_box = boxes[score_sort_index[0], :].view(-1, 4)
        res_box = boxes[score_sort_index[1], :].view(-1, 4)

        ious = utils.box_iou(cur_box, res_box).view(-1)

        inds = torch.nonzero(ious < threshold).squeeze()

        score_sort_index = score_sort_index[inds + 1].view(-1)

    return torch.LongTensor(keep)

def generate_prediction_boxes(box):
    box_ = box.clone()
    box_.squeeze_(0)
    feature_size = int(cfg.resize[0] / cfg.scale_size)
    scale_factor = 1 / feature_size

    for grid_y in range(box_.size(0)):
        for grid_x in range(box_.size(1)):
            box_[grid_y, grid_x, :, 0] = (box[0, grid_y, grid_x, :, 0] + grid_x) * scale_factor
            box_[grid_y, grid_x, :, 1] = (box[0, grid_y, grid_x, :, 1] + grid_y) * scale_factor
            box_[grid_y, grid_x, :, 2] = box[0, grid_y, grid_x, :, 2] * scale_factor
            box_[grid_y, grid_x, :, 3] = box[0, grid_y, grid_x, :, 3] * scale_factor

    box_ = box_.view(-1, 4)
    box_ = utils.xywh2xxyy(box_)
    box_ = torch.clamp(box_, min=0, max=1)
    return box_

def start_eval(conf_threshold=0.6, nms_threshold=0.3):

    args = parse_args()
    print('Called with args:')
    print(args)

    args.weights_file = os.path.join('weights', 'pretrained', 'darknet19.pth')

    if args.use_cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    if args.dataset == 'test':
        root = os.path.join(cfg.data_root, 'Images/Test/')
    elif args.dataset == 'temp':
        root = os.path.join(cfg.data_root, cfg.img_root)
    else:
        raise NotImplementedError

    if args.model == 'default':
        PATH = os.path.join(cfg.output_dir, 'yolov2_epoch_160.pth')
    else:
        PATH = os.path.join(cfg.output_dir, args.model)

    transform = A.Compose([
        A.Resize(cfg.resize[0], cfg.resize[1], p=1),
        A.Normalize(),
    ])

    test_dataset = VOCDataset(img_root=root, transform=transform, train=False)

    model = Yolo_v2(weights_file=args.weights_file)
    checkpoint = torch.load(PATH, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    for i in range(len(test_dataset)):
        transformed_img = test_dataset[i][0]
        image = test_dataset[i][1]

        img = torch.from_numpy(transformed_img)
        img = img.permute(2, 0, 1).contiguous()
        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        out = model(img).detach().cpu()
        bsize, _, H, W = out.size()
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, H, W, len(cfg.anchor_box), 25)

        pred_conf = torch.sigmoid(out[..., 20])
        pred_xy = torch.sigmoid(out[..., 21:23])
        pred_wh = torch.exp(out[..., 23:25])
        pred_cls = torch.nn.Softmax(dim=-1)(out[..., :20])
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_box = utils.generate_anchorbox(pred_box, device)

        pred_box = generate_prediction_boxes(pred_box)

        box_pred, conf_pred, max_conf, max_id = filter_boxes(pred_box, pred_conf, pred_cls, conf_threshold)

        detections = []

        max_id = max_id.view(-1)
        for cls in range(cfg.num_classes):
            cls_mask = max_id == cls
            inds = torch.nonzero(cls_mask).squeeze()

            if inds.numel() == 0:
                continue

            boxes_pred_class = box_pred[inds, :].view(-1, 4)
            conf_pred_class = conf_pred[inds, :].view(-1, 1)
            cls_max_conf_class = max_conf[inds].view(-1, 1)
            classes_class = max_id[inds].view(-1, 1)

            nms_keep = yolo_nms(boxes_pred_class, conf_pred_class.view(-1), nms_threshold)

            boxes_pred_class_keep = boxes_pred_class[nms_keep, :]
            conf_pred_class_keep = conf_pred_class[nms_keep, :]
            cls_max_conf_class_keep = cls_max_conf_class.view(-1, 1)[nms_keep, :]
            classes_class_keep = classes_class.view(-1, 1)[nms_keep, :]

            seq = [boxes_pred_class_keep, conf_pred_class_keep, cls_max_conf_class_keep, classes_class_keep.float()]

            detections_cls = torch.cat(seq, dim=-1)
            detections.append(detections_cls)
        res = torch.cat(detections, dim=0)
        visualize(image, res[:, :4], res[:, 6])
        # break


if __name__ == "__main__":
    start_eval(conf_threshold=0.6, nms_threshold=0.4)

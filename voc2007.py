import os
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import utils
import config as cfg
import albumentations as A
import cv2


class VOCDataset(Dataset):
    def __init__(self, data_root, train=True, transforms=None, resize=cfg.resize):
        self.annot_root = data_root + cfg.annot_root
        self.img_root = data_root + cfg.img_root
        self.train = train
        self.transforms = transforms
        self.resize_factor = resize

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

    def _check_exists(self):
        # print("Image Folder@~ {}".format(os.path.abspath(self.img_root)))
        # print("Label Folder@~ {}".format(os.path.abspath(self.annot_root)))

        return os.path.exists(self.img_root) and os.path.exists(self.annot_root)

    def __len__(self):
        return len(os.listdir(self.img_root))

    def __getitem__(self, idx):
        ann_files = os.listdir(self.annot_root)
        img_files = os.listdir(self.img_root)

        img_name = img_files[img_files.index("{}.jpg".format(ann_files[idx].split(".")[0]))]
        img_file = os.path.join(self.img_root, img_name)
        image = Image.open(img_file).convert("RGB")
        image = np.array(image)
        # image = cv2.imread(img_file)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.train:
            xml = open(os.path.join(self.annot_root, ann_files[idx]))
            tree = ET.parse(xml)
            root = tree.getroot()

            objs = root.findall("object")
            num_objs = len(objs)

            # gt = torch.zeros((num_objs, 5))
            bboxes = []
            clses = []
            for i, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text)
                y1 = int(bbox.find('ymin').text)
                x2 = int(bbox.find('xmax').text)
                y2 = int(bbox.find('ymax').text)

                # cls = classes.index(obj.find('name').text)
                cls = obj.find('name').text
                # gt[i, :] = torch.LongTensor([x1, y1, x2, y2, cls])
                bboxes.append([x1, y1, x2, y2])
                clses.append(cls)

            if self.transforms:
                transformed = self.transforms(image=image, bboxes=bboxes, class_labels=clses)
                print(transformed)
                return transformed["image"], transformed["bboxes"], transformed["class_labels"]
            else:
                return image, bboxes, clses
        else:
            if self.transforms:
                transformed = self.transforms(image=image)
                return transformed["image"]
            else:
                return image


def detection_collate(batch):
    imgs = []
    targets = []

    feature_size = cfg.feature_size
    num_classes = cfg.num_classes

    for sample in batch:

        imgs.append(sample[0])
        label = torch.zeros((feature_size, feature_size, (num_classes + 5)))

        boxes = sample[1]
        gt_classes = sample[2]
        gt_classes.unsqueeze_(1)

        num_obj = boxes.size(0)
        objectness = torch.ones((num_obj, 1))
        scale_factor = 1 / feature_size
        grid_index = boxes[:, 0:2] // scale_factor
        offset = boxes[:, 0:2] / scale_factor - grid_index
        gt_box = torch.cat([objectness, offset, boxes[:, 2:4]], dim=1)

        label[grid_index[:, 0].long(), grid_index[:, 1].long(), num_classes:num_classes+5] = gt_box
        label[grid_index[:, 0].long(), grid_index[:, 1].long(), gt_classes[:, 0].long()] = 1

        targets.append(label)

    return torch.stack(imgs, dim=0), torch.stack(targets, dim=0)



if __name__ == '__main__':
    from visualize import *

    transforms = A.Compose([
        # A.RandomResizedCrop(cfg.resize, cfg.resize, scale=(0.08, 1.0), ratio=(0.75, 1.33), p=0.5),
        # A.RandomSizedBBoxSafeCrop(cfg.resize, cfg.resize, p=1),
        A.RandomCrop(300, 300, p=1),
        A.Resize(cfg.resize, cfg.resize, p=1),
        A.RandomBrightnessContrast(),
        A.HueSaturationValue(),
    ], bbox_params=A.BboxParams(format='pascal_voc', min_visibility=0.3, label_fields=['class_labels']))

    data = VOCDataset(data_root=cfg.data_root,
                      train=True,
                      transforms=transforms)

    data[0]
    # for b in range(cfg.batch_size):
    #     img = data[b][0]
    #     bboxes = data[b][1]
    #     clses = data[b][2]
        # visualize(img, bboxes, clses)
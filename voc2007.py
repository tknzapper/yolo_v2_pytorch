import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import utils
import config as cfg
import albumentations as A


class VOCDataset(Dataset):
    def __init__(self, img_root, transform, train=True):
        self.annot_root = cfg.data_root + cfg.annot_root
        self.img_root = img_root
        self.transform = transform
        self.train = train

        if not self._check_exists():
            raise RuntimeError("Dataset not found.")

    def _check_exists(self):
        # print("Image Folder@~ {}".format(os.path.abspath(self.img_root)))
        # print("Label Folder@~ {}".format(os.path.abspath(self.annot_root)))
        if self.train:
            return os.path.exists(self.img_root) and os.path.exists(self.annot_root)
        else:
            return os.path.exists(self.img_root)

    def __len__(self):
        return len(os.listdir(self.img_root))

    def __getitem__(self, idx):
        img_files = os.listdir(self.img_root)
        img_file = os.path.join(self.img_root, img_files[idx])
        image = np.array(Image.open(img_file).convert("RGB"))

        if self.train:
            ann_file = img_files[idx].split('.')[-2] + '.xml'
            xml = open(os.path.join(self.annot_root, ann_file))
            tree = ET.parse(xml)
            root = tree.getroot()

            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            # print(width, height)

            objs = root.findall("object")
            num_objs = len(objs)
            bbox = torch.zeros((num_objs, 4))
            classes = torch.zeros((num_objs))
            for i, obj in enumerate(objs):
                box = obj.find('bndbox')
                x1 = int(box.find('xmin').text) + 1
                y1 = int(box.find('ymin').text) + 1
                x2 = int(box.find('xmax').text) - 1
                y2 = int(box.find('ymax').text) - 1
                cls = cfg.classes.index(obj.find('name').text)
                bbox[i, :] = torch.FloatTensor([x1/width, y1/height, x2/width, y2/height])
                classes[i] = cls
            utils.xxyy2xywh(bbox)
            transformed = self.transform(image=image, bboxes=bbox, class_labels=classes)
            return transformed["image"], transformed["bboxes"], transformed["class_labels"]
        else:
            transformed = self.transform(image=image)
            return transformed["image"]


def detection_collate(batch):
    imgs = []
    targets = []

    feature_size = cfg.feature_size
    num_classes = cfg.num_classes

    for sample in batch:
        img = torch.from_numpy(sample[0])
        img = img.permute(2, 0, 1).contiguous()
        imgs.append(img)

        label = torch.zeros((feature_size, feature_size, (num_classes + 5)))
        bboxes = sample[1]
        classes = sample[2]
        num_obj = len(bboxes)
        for objs in range(num_obj):
            bbox = bboxes[objs]
            cls = classes[objs].long()
            objectness = 1
            scale_factor = 1 / feature_size
            grid_x_idx = int(bbox[0] // scale_factor)
            grid_y_idx = int(bbox[1] // scale_factor)
            x_offset = bbox[0] / scale_factor - grid_x_idx
            y_offset = bbox[1] / scale_factor - grid_y_idx
            w = bbox[2] / scale_factor
            h = bbox[3] / scale_factor
            label[grid_y_idx, grid_x_idx, num_classes:num_classes+5] = torch.FloatTensor([objectness, x_offset, y_offset, w, h])
            label[grid_y_idx, grid_x_idx, cls] = 1
        targets.append(label)

    return torch.stack(imgs, dim=0), torch.stack(targets, dim=0)



if __name__ == '__main__':
    from visualize import *
    from torch.utils.data import DataLoader

    transform = A.Compose([
        # A.RandomSizedBBoxSafeCrop(cfg.resize, cfg.resize, erosion_rate=0.2, p=1),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Resize(cfg.resize, cfg.resize, p=1),
        A.Normalize(),
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    train_root = os.path.join(cfg.data_root, 'Images/Test/')

    train_dataset = VOCDataset(img_root=train_root,
                               transform=transform)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=len(train_dataset),
                              shuffle=False,
                              collate_fn=detection_collate)

    # train_dataset[0]

    # for b in range(cfg.batch_size):
    #     img = train_dataset[b][0]
        # bboxes = train_dataset[b][1]
        # classes = train_dataset[b][2]
        # visualize(img, bboxes, classes)
        # print(img)

    for i, data in enumerate(train_loader, 0):
        inputs, targets = data

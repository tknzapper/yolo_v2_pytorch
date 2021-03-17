from config import *
import os
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils import *


class VOCDataset(Dataset):
    def __init__(self, data_root, train=True, transforms=None, resize=resize):
        self.annot_root = data_root + annot_root
        self.img_root = data_root + img_root
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
        # current_shape = np.asarray(image)
        # current_shape = torch.tensor(current_shape.shape)
        image = image.resize((self.resize_factor, self.resize_factor))
        image = np.array(image) / 255.
        image = image.reshape(3, self.resize_factor, self.resize_factor)
        image = torch.from_numpy(image)

        if self.train:
            xml = open(os.path.join(self.annot_root, ann_files[idx]))
            tree = ET.parse(xml)
            root = tree.getroot()

            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)
            # channels = int(size.find("depth").text)

            objs = root.findall("object")
            num_objs = len(objs)

            boxes = np.zeros((num_objs, 4), dtype=np.float16)
            gt_classes = np.zeros((num_objs), dtype=np.int8)

            for i, obj in enumerate(objs):
                bbox = obj.find('bndbox')
                x1 = int(bbox.find('xmin').text) / width
                y1 = int(bbox.find('ymin').text) / height
                x2 = int(bbox.find('xmax').text) / width
                y2 = int(bbox.find('ymax').text) / height

                cls = classes.index(obj.find('name').text)

                boxes[i, :] = [x1, y1, x2, y2]
                gt_classes[i] = cls

            boxes = xxyy2xywh(torch.FloatTensor(boxes))
            # print(img_name)

            return image, boxes, gt_classes
        else:
            return image


def detection_collate(batch):
    imgs = []
    targets = []

    label = torch.zeros((feature_size, feature_size, (num_classes + 5) * len(anchor_box)))
    for sample in batch:
        imgs.append(sample[0])

        boxes = sample[1]
        gt_classes = sample[2]

        scale_factor = 1 / feature_size
        bsize, isize = boxes[:, 0:2].size()
        objectness = torch.zeros((bsize, 1))
        grid_index = torch.zeros((bsize, isize))
        offset = torch.zeros((bsize, isize))
        for b in range(bsize):
            for i in range(isize):
                objectness[b] = 1
                g_index = int(float(boxes[:, 0:2][b, i].item()) // scale_factor)
                offs = float(boxes[b, i].item() / scale_factor) - g_index
                grid_index[b, i] = g_index
                offset[b, i] = offs
        grid_arr = torch.cat([objectness, offset, boxes[:, 2:4]], dim=1)
        # print(grid_arr)

        b_size, _ = grid_index.size()
        for b in range(b_size):
            for anchor in range(len(anchor_box)):
                idx = (num_classes + 5) * (anchor + 1)
                label[int(grid_index[b][0].item()), int(grid_index[b][1].item()), idx-5:idx] = grid_arr[b]
                label[int(grid_index[b][0]), int(grid_index[b][1]), gt_classes[b]+idx-25] = 1

        targets.append(label)

    return torch.stack(imgs, dim=0), torch.stack(targets, dim=0)



if __name__ == '__main__':

    data = VOCDataset(data_root, train=True)
    print(data[0])

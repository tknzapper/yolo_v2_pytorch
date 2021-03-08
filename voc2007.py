from config import *
import os
import numpy as np
import random
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset


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
        # print("Image Folder : {}".format(self.img_root))
        # print("Label Folder : {}".format(self.annot_root))

        return os.path.exists(self.img_root) and os.path.exists(self.annot_root)

    def __len__(self):
        return len(os.listdir(self.img_root))

    def __getitem__(self, idx):
        ann_files = os.listdir(self.annot_root)
        img_files = os.listdir(self.img_root)

        img_name = img_files[img_files.index("{}.jpg".format(ann_files[idx].split(".")[0]))]
        img_file = os.path.join(self.img_root, img_name)
        image = Image.open(img_file).convert("RGB")
        current_shape = np.asarray(image)
        current_shape = torch.tensor(current_shape.shape)
        image = image.resize((self.resize_factor, self.resize_factor))
        image = np.array(image) / 255.
        image = image.reshape(3, self.resize_factor, self.resize_factor)
        image = torch.from_numpy(image)

        if not self.train:
            xml = open(os.path.join(self.annot_root, ann_files[idx]))
            tree = ET.parse(xml)
            root = tree.getroot()

            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            channels = int(size.find("depth").text)

            objects = root.findall("object")
            label = []
            for _object in objects:
                name = _object.find("name").text
                try:
                    _class = classes.index(name)
                    bndbox = _object.find("bndbox")
                    xmin = int(bndbox.find("xmin").text) / float(width)
                    ymin = int(bndbox.find("ymin").text) / float(height)
                    xmax = int(bndbox.find("xmax").text) / float(width)
                    ymax = int(bndbox.find("ymax").text) / float(height)
                    bbox = self.transcoord((xmin, ymin, xmax, ymax))
                    label.append((_class, bbox[0], bbox[1], bbox[2], bbox[3]))
                except:
                    continue
            return image, label, current_shape
        else:
            return image

    def transcoord(self, bndbox):
        cx = float((bndbox[0] + bndbox[2]) / 2.)
        cy = float((bndbox[1] + bndbox[3]) / 2.)
        w = float(bndbox[2] - bndbox[0])
        h = float(bndbox[3] - bndbox[1])

        return cx, cy, w, h


def detection_collate(batch):
    targets = []
    imgs = []
    sizes = []

    for sample in batch:
        imgs.append(sample[0])
        sizes.append(sample[2])

        label = torch.zeros((feature_size, feature_size, (num_classes + 5) * len(anchor_box)))
        for object in sample[1]:
            objecness = 1
            class_idx = object[0]

            cx = object[1]
            cy = object[2]
            w = object[3]
            h = object[4]



if __name__ == '__main__':
    data = VOCDataset(data_root)
    print(data[0])
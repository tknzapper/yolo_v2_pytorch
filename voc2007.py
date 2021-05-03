import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
import utils
import config as cfg
import albumentations as A


transform = A.Compose([
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        # A.Resize(cfg.resize[0], cfg.resize[1], p=1),
        A.Normalize(),
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))


class VOCDataset(Dataset):
    def __init__(self, image_set, year, transform=transform, train=True):
        self.image_set = image_set
        self.year = year
        self.image_idx = self._load_image_set_idx()
        self.transform = transform
        self.train = train

    def __len__(self):
        return len(self.image_idx)

    def _get_default_path(self):
        root_dir = os.path.join(os.path.dirname(__file__))
        data_dir = os.path.join(root_dir, 'data', 'VOCdevkit')
        return os.path.join(data_dir, 'VOC' + self.year)

    def _load_image_set_idx(self):
        image_set_file = os.path.join(self._get_default_path(),
                                      'ImageSets', 'Main', self.image_set + '.txt')
        assert os.path.exists(image_set_file), \
            'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_idx = [x.strip() for x in f.readlines()]
        return image_idx

    def _image_path_at(self, idx):
        image_root = os.path.join(self._get_default_path(), 'JPEGImages')
        return os.path.join(image_root, self._image_path_from_idx(idx))

    def _image_path_from_idx(self, idx):
        return self.image_idx[idx] + '.jpg'

    def _annotation_path_at(self, idx):
        annotation_root = os.path.join(self._get_default_path(), 'Annotations')
        return os.path.join(annotation_root, self._annotation_path_from_idx(idx))

    def _annotation_path_from_idx(self, idx):
        return self.image_idx[idx] + '.xml'

    def __getitem__(self, idx):
        image = Image.open(self._image_path_at(idx)).convert('RGB')
        img_width, img_height = image.width, image.height
        image = image.resize(cfg.resize)
        image = np.array(image)

        if self.train:
            xml = open(self._annotation_path_at(idx))
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
                x1 = int(box.find('xmin').text) - 1
                y1 = int(box.find('ymin').text) - 1
                x2 = int(box.find('xmax').text) - 1
                y2 = int(box.find('ymax').text) - 1
                cls = cfg.classes.index(obj.find('name').text)
                bbox[i, :] = torch.FloatTensor([x1/width, y1/height, x2/width, y2/height])
                classes[i] = cls
            bbox = utils.xxyy2xywh(bbox)
            transformed = self.transform(image=image, bboxes=bbox, class_labels=classes)
            return transformed["image"], transformed["bboxes"], transformed["class_labels"]
        else:
            transformed = self.transform(image=image)
            return transformed["image"], self._image_path_from_idx(idx)


def detection_collate(batch):
    imgs = []
    targets = []

    # feature_size = cfg.feature_size
    num_classes = cfg.num_classes

    for sample in batch:
        img = torch.from_numpy(sample[0])
        img = img.permute(2, 0, 1).contiguous()
        imgs.append(img)

        feature_size = int(img.size(1) / cfg.scale_size)
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

    db07 = VOCDataset(image_set='trainval', year='2007')
    db12 = VOCDataset(image_set='trainval', year='2012')
    dataset = db07 + db12
    print(dataset[0])
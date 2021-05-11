import os
import numpy as np
import xml.etree.ElementTree as ET
from PIL import Image
import torch
from torch.utils.data import Dataset
from utils.bbox import xxyy2xywh
from config import config as cfg
import albumentations as A


transform = A.Compose([
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
    A.HueSaturationValue(10, 10, 10, p=0.5),
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
        image = image.resize(cfg.resize)
        image = np.array(image)

        if self.train:
            xml = open(self._annotation_path_at(idx))
            tree = ET.parse(xml)
            root = tree.getroot()

            size = root.find("size")
            width = float(size.find("width").text)
            height = float(size.find("height").text)

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
            bbox = xxyy2xywh(bbox)
            num_obj = torch.Tensor([bbox.size(0)]).long()
            transformed = self.transform(image=image, bboxes=bbox, class_labels=classes)
            return torch.FloatTensor(transformed["image"]).permute(2, 0, 1).contiguous(), transformed["bboxes"], transformed["class_labels"], num_obj
        else:
            transformed = self.transform(image=image)
            return transformed["image"], self._image_path_from_idx(idx)


def detection_collate(batch):
    bsize = len(batch)
    img, bbox, cls, num_obj = zip(*batch)

    max_num_obj = max([x.item() for x in num_obj])
    padded_box = torch.zeros((bsize, max_num_obj, 4))
    padded_cls = torch.zeros((bsize, max_num_obj))

    for b in range(bsize):
        padded_box[b, :num_obj[b], :] = bbox[b]
        padded_cls[b, :num_obj[b]] = cls[b]

    return torch.stack(img, dim=0), padded_box, padded_cls, torch.stack(num_obj, dim=0)


if __name__ == '__main__':

    db07 = VOCDataset(image_set='train', year='2007')
    db12 = VOCDataset(image_set='trainval', year='2012')
    dataset = db07 + db12
    print(dataset[0])

    # import shutil
    #
    # test = VOCDataset(image_set='test', year='2007')
    # train07 = VOCDataset(image_set='train', year='2007')
    # train12 = VOCDataset(image_set='train', year='2012')
    #
    # dataset = train12
    # total = len(dataset.image_idx)
    # annot_dst = './data/annotations/'
    # img_dst = './data/images/'
    # for idx in range(total):
    #     annot_src = dataset._annotation_path_at(idx)
    #     img_src = dataset._image_path_at(idx)
    #     shutil.copy(annot_src, annot_dst)
    #     shutil.copy(img_src, img_dst)
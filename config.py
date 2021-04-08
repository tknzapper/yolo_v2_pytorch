data_root = './data/VOCdevkit/VOC2007/'
img_root = 'JPEGimages/'
# img_root = 'temp_image/'
annot_root = 'Annotations/'
# annot_root = 'temp_label/'

classes = ['person',
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

num_classes = len(classes) # 20
feature_size = 13

anchor_box = [(1.3221, 1.73145),
              (3.19275, 4.00944),
              (5.05587, 8.09892),
              (9.47112, 4.84053),
              (11.2364, 10.0071)]

resize = 416
batch_size = 1

lambda_coord = 5
lambda_noobj = 0.5

pw = int(resize / feature_size)
ph = int(resize / feature_size)

font = './fonts/d2coding.ttf'

layer1 = [
    (32, 3, 1, 1),        # (out_channels, kernel_size)
    "M",
    (64, 3, 1, 1),
    "M",
    (128, 3, 1, 1),
    (64, 1, 1, 0),
    (128, 3, 1, 1),
    "M",
    (256, 3, 1, 1),
    (128, 1, 1, 0),
    (256, 3, 1, 1),
    "M",
    (512, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),
    (256, 1, 1, 0),
    (512, 3, 1, 1),         # reorg layer
]

layer2 = [
    "M",
    (1024, 3, 1, 1),
    (512, 1, 1, 0),
    (1024, 3, 1, 1),
    (512, 1, 1, 0),
    (1024, 3, 1, 1)
]
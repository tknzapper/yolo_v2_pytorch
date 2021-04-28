data_root = './data/VOCdevkit/VOC2007/'
# img_root = 'JPEGimages/'
img_root = 'temp_image/'
annot_root = 'Annotations/'
output_dir = 'weights/model/'

font = './fonts/arial.ttf'

classes = ['person',
           'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'diningtable', 'pottedplant', 'sofa', 'tvmonitor']

num_classes = len(classes) # 20

anchor_box = [[1.3221,  1.73145],
              [3.19275, 4.00944],
              [5.05587, 8.09892],
              [9.47112, 4.84053],
              [11.2364, 10.0071]]

batch_size = 16
epochs = 160
lr = 1e-4
lr_decay = [60, 90]

lambda_coord = 5
lambda_obj = 1
lambda_noobj = 1
lambda_cls = 1

multi_scale = True

scale_step = 40

scale_range = (3, 4)

epoch_scale = {
    1:  (3, 4),
    15: (2, 5),
    60: (1, 6),
    90: (0, 7),
}

scale_size = 32
resize = (448, 448)

input_sizes = [(224, 224), # 7 x 7
               (288, 288), # 9 x 9
               (352, 352), # 11 x 11
               (416, 416), # 13 x 13
               (480, 480), # 15 x 15
               (544, 544), # 17 x 17
               (608, 608)] # 19 x 19


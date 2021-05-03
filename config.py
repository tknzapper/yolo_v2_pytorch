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

batch_size = 12
epochs = 160
lr = 1e-4
lr_decay = [60, 90]

lambda_coord = 5
lambda_obj = 1
lambda_noobj = 1
lambda_cls = 1

multi_scale = True
scale_step = 50
scale_range = (3, 4)
epoch_scale = {
    1:   (1, 2),
    15:  (0, 3),
    60:  (0, 4),
    90:  (0, 5),
}

scale_size = 32
resize = (416, 416)
input_sizes = [(352, 352), # 0: 11 x 11
               (416, 416), # 1: 13 x 13
               (480, 480), # 2: 15 x 15
               (544, 544), # 3: 17 x 17
               (608, 608)] # 4: 19 x 19

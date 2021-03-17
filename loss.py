from model import *
from utils import *
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='sum')
        self.feature_size = feature_size
        self.anchor = torch.FloatTensor(anchor_box)
        self.num_classes = num_classes

    def forward(self, prediction, target):
        bsize, _, h, w = prediction.size()
        out = prediction.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * len(self.anchor), 5 + self.num_classes)

        B, H, W, _ = target.size()
        target = target.view(B, H * W * len(self.anchor), 5 + self.num_classes).contiguous()

        # sorting prediction data
        pred_conf = torch.sigmoid(out[:, :, 20])
        pred_xy = torch.sigmoid(out[:, :, 21:23])
        pred_wh = torch.exp(out[:, :, 23:25])
        pred_cls = out[:, :, :20]
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_data = (pred_box, pred_conf, pred_cls)

        # sorting target data
        gt_conf = target[:, :, 20]
        gt_xy = target[:, :, 21:23]
        gt_wh = target[:, :, 23:25]
        gt_cls = target[:, :, :20]
        gt_box = torch.cat([gt_xy, gt_wh], dim=-1)
        gt_data = (gt_box, gt_conf, gt_cls)

        self._calc_loss(pred_data, gt_data, h, w)


    def _calc_loss(self, pred_data, gt_data, h, w):
        pred_box = pred_data[0]
        pred_conf = pred_data[1]
        pred_cls = pred_data[2]

        gt_box = gt_data[0]
        gt_conf = gt_data[1]
        gt_cls = gt_data[2]

        # calculating iou

        bsize = pred_box.size(0)
        pred_box = pred_box.view(bsize, h * w, len(self.anchor), -1)
        gt_box = gt_box.view(bsize, h * w, len(self.anchor), -1)
        anc_box = generate_anchorbox(pred_box)

        gt_conf = gt_conf.view(bsize, h * w, len(self.anchor), -1)
        mask = gt_conf.new_zeros(bsize, h * w, len(self.anchor), 1)
        iou_target = gt_conf.new_zeros(bsize, h * w, len(self.anchor), 1)

        iou = box_iou(anc_box, gt_box)
        print(iou.shape)
        max_iou, anchor_idx = torch.max(iou, dim=-1, keepdim=True)
        max_iou = max_iou.view(bsize, h * w, -1)
        anchor_idx = anchor_idx.view(bsize, -1)
        # print(max_iou.shape)

        mask[:, :, anchor_idx, :] = gt_conf[:, :, anchor_idx, :]
        iou_target[:, :, anchor_idx, :] = max_iou
        # print(iou_target)
        # print(torch.nonzero(gt_conf))
        # print(mask.view(bsize, h, w, -1))

        box_loss = 1 / batch_size * lambda_coord * self.mse(pred_box * mask, gt_box * mask)
        # iou_loss = 1 / batch_size * self.mse(pred_conf * mask, )



if __name__ == "__main__":
    from voc2007 import *
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms


    train_dataset = VOCDataset(data_root=data_root,
                               transforms=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=detection_collate)
    model = Yolo_v2().to(device)
    criterion = Loss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-6)

    epochs = 50
    for epoch in range(epochs):
        running_loss = 0.0

        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs = inputs.float().to(device)
            labels = labels.float().to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # print(outputs)
            loss = criterion(outputs, labels)
            print(loss)

            break
        break
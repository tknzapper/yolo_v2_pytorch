from model import *
from utils import *

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')
        self.feature_size = feature_size
        self.anchor = torch.FloatTensor(anchor_box)
        self.num_classes = num_classes

    def forward(self, prediction, target):
        bsize, _, h, w = prediction.size()
        out = prediction.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * len(self.anchor), 5 + self.num_classes)

        B, H, W, _ = target.size()
        target = target.view(B, H * W * len(self.anchor), 5 + self.num_classes).contiguous()

        # sorting prediction data
        pred_conf = torch.sigmoid(out[:, :, 20]) # pred_bbox, gt_bbox iou로 계산
        pred_xy = torch.sigmoid(out[:, :, 21:23])
        pred_wh = torch.exp(out[:, :, 23:25])
        pred_wh = pred_wh.view(bsize, h * w, len(self.anchor), -1)
        for anchor in range(len(self.anchor)):
            pred_wh[:, :, anchor, 0] *= self.anchor[anchor, 0]
            pred_wh[:, :, anchor, 1] *= self.anchor[anchor, 1]
        pred_wh = pred_wh.view(bsize, h * w * len(self.anchor), -1)
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
        # print(gt_box.shape)

        self._calc_loss(pred_data, gt_data, h, w)


    def _calc_loss(self, pred_data, gt_data, h, w):
        pred_box = pred_data[0]
        pred_conf = pred_data[1]
        pred_cls = pred_data[2]

        gt_box = gt_data[0]
        gt_conf = gt_data[1]
        gt_cls = gt_data[2]

        # ================= #
        #  Calculating IoU  #
        # ================= #

        bsize = pred_box.size(0)
        pred_box = pred_box.view(bsize, h * w, len(self.anchor), -1)
        gt_box = gt_box.view(bsize, h * w, len(self.anchor), -1)

        iou = box_iou(gt_box, gt_box)
        max_iou, anchor_idx = torch.max(iou, dim=-1, keepdim=True)
        max_iou = max_iou.view(bsize, -1)
        anchor_idx = anchor_idx.view(bsize, -1)
        print(max_iou)
        print(anchor_idx)
        threshold = 0.5
        iou_thresh_filter = max_iou.view(-1) >= threshold
        mask = torch.where(iou_thresh_filter.view(-1, h * w))
        # print(mask[0].item(), mask[1].item())
        print(mask)
        iou_mask = torch.zeros((bsize, h * w, len(self.anchor), 1))
        for b in range(bsize):
            for g in range(h * w):
                # print(mask[b][g])
                # print(anchor_idx[b, g].item())
                iou_mask[b, g, anchor_idx[b, g].item(), :] = lambda_coord
        # print(iou_mask.shape)


        # print(gt_conf.view(bsize, h * w, len(self.anchor)))


        # ================= #
        # Localization Loss #
        # ================= #

        # xy_coord loss



if __name__ == "__main__":
    from voc2007 import *
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = VOCDataset(data_root=data_root,
                               transforms=transforms.ToTensor())

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=detection_collate)
    model = Yolo_v2().to(device)
    criterion = Loss()
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
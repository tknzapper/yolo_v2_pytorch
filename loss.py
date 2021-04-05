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
        target = target.view(bsize, h * w, 5 + self.num_classes).contiguous()

        # sorting prediction data
        pred_conf = torch.sigmoid(out[:, :, 20:21])
        pred_xy = torch.sigmoid(out[:, :, 21:23])
        pred_wh = torch.exp(out[:, :, 23:25])
        pred_cls = out[:, :, :20]
        pred_box = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_data = (pred_box, pred_conf, pred_cls)

        # sorting target data
        gt_conf = target[:, :, 20:21]
        gt_xy = target[:, :, 21:23]
        gt_wh = target[:, :, 23:25]
        gt_cls = target[:, :, :20]
        gt_box = torch.cat([gt_xy, gt_wh], dim=-1)
        gt_data = (gt_box, gt_conf, gt_cls)


        return self._calc_loss(pred_data, gt_data, h, w)


    def _calc_loss(self, pred_data, gt_data, h, w):
        pred_box = pred_data[0]
        pred_conf = pred_data[1]
        pred_cls = pred_data[2]

        gt_box = gt_data[0]
        gt_conf = gt_data[1]
        gt_cls = gt_data[2]

        bsize = pred_box.size(0)
        pred_box = pred_box.view(bsize, h * w, len(self.anchor), -1)
        gt_box = gt_box.view(bsize, h * w, 1, -1)
        anc_box = generate_anchorbox(pred_box)
        iou = box_iou(anc_box, gt_box)

        max_iou, anchor_idx = torch.max(iou, dim=-1, keepdim=True)
        max_iou = max_iou.view(bsize, h * w, -1)
        anchor_idx = anchor_idx.view(bsize, h * w, -1)

        idx = torch.nonzero(gt_conf.view(bsize, h * w))
        batch = idx[:, 0].long()
        cell_idx = idx[:, 1].long()
        trans_idx = idx.T
        batch.unsqueeze_(1)
        cell_idx.unsqueeze_(1)
        argmax_anchor_idx = anchor_idx[batch, cell_idx].squeeze(2)

        mask = pred_box.new_zeros(bsize, h * w, len(self.anchor), 1)
        mask[batch, cell_idx, argmax_anchor_idx] = gt_conf[batch, cell_idx]
        mask = mask.int() >= 1

        pred_conf = pred_conf.view(bsize, h * w, len(self.anchor), -1)
        target_conf = pred_box.new_zeros(bsize, h * w, len(self.anchor), 1)
        target_conf[batch, cell_idx, argmax_anchor_idx] = 1.

        pred_cls = pred_cls.view(bsize, h * w, len(self.anchor), -1)
        pred_cls = pred_cls[trans_idx[0], trans_idx[1], anchor_idx[trans_idx[0], trans_idx[1]][0]]
        gt_cls = gt_cls[trans_idx[0], trans_idx[1]]
        gt_cls = torch.max(gt_cls, dim=1)[1].long()

        box_loss = lambda_coord * self.mse(pred_box * mask, gt_box * mask)
        conf_loss = self.mse(pred_conf * mask, target_conf * mask)
        noobj_loss = lambda_noobj * self.mse(pred_conf * ~mask, target_conf * ~mask)
        cls_loss = self.cross_entropy(pred_cls, gt_cls).mean()
        total_loss = box_loss + conf_loss + noobj_loss + cls_loss

        print("box_loss: {}".format(box_loss))
        print("conf_loss: {}".format(conf_loss))
        print("noobj_loss: {}".format(noobj_loss))
        print("cls_loss: {}".format(cls_loss))

        print()
        print()

        return total_loss



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
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            print(loss)

            # loss.backward()
            # optimizer.step()

            break
        break

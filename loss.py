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
        batch.unsqueeze_(1)
        cell_idx.unsqueeze_(1)

        argmax_anchor_idx = anchor_idx[batch, cell_idx].squeeze(2)
        # print(argmax_anchor_idx)
        # print(argmax_anchor_idx)
        mask = pred_box.new_zeros(bsize, h * w, len(self.anchor), 1)
        cls_mask = pred_box.new_zeros(bsize, h * w, 1)
        target_conf = pred_box.new_zeros(bsize, h * w, len(self.anchor), 1)


        mask[batch, cell_idx, argmax_anchor_idx] = gt_conf[batch, cell_idx]

        mask = mask.int() >= 1
        cls_mask = gt_conf[batch, cell_idx].int() >= 1

        # print(mask[:, 85, :, :])
        # print(mask[:, 71, :, :])
        #
        # not_mask = mask >= 1
        #
        # print(~not_mask[:, 85, :, :])
        # print(~not_mask[:, 71, :, :])
        #target_conf[batch, cell_idx, argmax_anchor_idx] = max_iou[batch, cell_idx]
        target_conf[batch, cell_idx, argmax_anchor_idx] = 1.
        # print(mask[batch[1], cell_idx[1], argmax_anchor_idx[1]])
        pred_conf = pred_conf.view(bsize, h * w, len(self.anchor), -1)
        pred_cls = pred_cls.view(bsize, h * w, len(self.anchor), -1)

        # print("pred_cls Pre : ", pred_cls.shape)
        # print("gt_cls Pre : ", gt_cls.shape)
        # print("anchor_idx Pre : ", anchor_idx.shape)


        # mask 1 : GT Confidence가 1인 예들만
        # mask 2 : Predict IoU가 가장 높은 Anchor Index
        #print(idx)
        #print(idx.shape)
        #gt_cls = gt_cls[idx]

        trans_idx = idx.T
        gt_cls = gt_cls[trans_idx[0], trans_idx[1]]
        gt_cls = torch.max(gt_cls, dim=1)[1].long()

        #gt_cls = torch.LongTensor(torch.max(gt_cls, dim=1)[0].cuda())
        #print(gt_cls)
        cls_keep = torch.nonzero(mask)
        #print(mask.shape)
        pred_cls = pred_cls[trans_idx[0], trans_idx[1], anchor_idx[trans_idx[0], trans_idx[1]][0]]
        # gt_cls = pred_cls[cls_keep]
        # print(pred_cls)
        # print(gt_cls)
        #print("pred_cls After : ", pred_cls.shape)
        #print("gt_cls After : ", gt_cls.shape)
        # print("anchor_idx After : ", anchor_idx[idx.T[0], idx.T[1]].shape)
        # pred_cls = pred_cls.view(-1, self.num_classes)
        # gt_cls = gt_cls.view(-1, self.num_classes)

        cls_keep = torch.nonzero(mask)
        #pred_cls = pred_cls[cls_keep]
        #gt_cls = gt_cls[cls_keep]

        # pred_cls = torch.argmax(pred_cls, 1)
        # gt_cls = torch.argmax(gt_cls, 1)
        #print(pred_cls.shape)
        #print(gt_cls.shape)

        # print(cls_keep)
        # print(cls_keep.shape)

        #print("pred_cls : ", pred_cls)
        #print("gt_cls : ", gt_cls)
        # print(pred_conf.shape)
        # print(mask.shape)
        # print(target_conf.shape)
        #
        # print(pred_conf * mask)
        print((mask > 0).nonzero(as_tuple=True))
        print((mask * pred_conf)[0, 85])
        print((mask * pred_conf)[1, 71])

        print((mask * target_conf)[0, 85])
        print((mask * target_conf)[1, 71])

        # print((pred_conf * mask > 0).nonzero(as_tuple=True))
        # print((target_conf * mask > 0).nonzero(as_tuple=True))
        #print(target_conf * mask)

        #box_loss = 1 / batch_size * lambda_coord * self.mse(pred_box * mask, gt_box * mask)
        #conf_loss = 1 / batch_size * self.mse(pred_conf * mask, target_conf * mask)
        #noobj_loss = 1 / batch_size * lambda_noobj * self.mse(pred_conf * ~mask, target_conf * ~mask)

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

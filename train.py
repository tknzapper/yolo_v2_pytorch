from voc2007 import *
from torch.utils.data import DataLoader
from loss import *
import torchvision.transforms as transforms
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



train_dataset = VOCDataset(data_root=data_root,
                           transforms=transforms.ToTensor())

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=False,
                          collate_fn=detection_collate)
model = Yolo_v2().to(device)
criterion = Loss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 1000
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
        #print(loss)

        loss.backward()
        optimizer.step()


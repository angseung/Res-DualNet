import torch
from torchvision import models

model = models.resnet18(num_classes=10)
model = torch.nn.DataParallel(model)
optim = torch.optim.Adam(model.parameters())

path = "outputs/resnet18_32/0_epoch.pth"
SAVEDAT = torch.load(path)

# model.load_state_dict(SAVEDAT['state_dict'])
optim.load_state_dict(SAVEDAT['optimizer'])

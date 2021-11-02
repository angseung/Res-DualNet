import torch
import torchinfo
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torchinfo import summary
from models.resnetCA_ori import ResDaulNet18_TPI5
from models.resnetCA import ResDaulNet18_TP5
from utils import data_loader, progress_bar
import math

# Check use GPU or not
use_gpu = torch.cuda.is_available()  # use GPU

if use_gpu:
    device = torch.device("cuda")
else:
    raise NotImplementedError('CUDA Device needed to run this code...')

# ImageNet
net = ResDaulNet18_TPI5()

# CIFAR-10
# net = ResDaulNet18_TP5()

net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

# Load checkpoint data
# {net : net.state_dict,
#  acc : best test acc,
#  optimizer : optimizer.state_dict(),
#  epoch : best performed epoch}

pth_path = "./outputs/resdual5_imagenet/ckpt.pth"
SAVEDAT = torch.load(pth_path)

net.load_state_dict(SAVEDAT['net'])
acc = SAVEDAT['acc']
epoch = SAVEDAT['epoch']

model_name = net.module.__class__.__name__
print("%s model was loaded successfully... [best validation acc : %.3f at %03d epoch]"
      %(model_name, acc, epoch))

mode = 'test'

input_size = 224
dataset = "ImageNet"

# input_size = 32
# dataset = "CIFAR-10"

batch_size = 100

dataloader = data_loader(
    mode=mode,
    dataset=dataset,
    input_size=input_size,
    batch_size=batch_size,
    shuffle_opt=True
)

print("Loading %s dataset completed..." % mode)

# Get model params and macs...
modelinfo = summary(net, (1, 3, input_size, input_size), verbose=0)
total_params = modelinfo.total_params
total_macs = modelinfo.total_mult_adds

param_mil = total_params / (10 ** 6)
macs_bil = total_macs / (10 ** 9)

if mode == 'train':
    raise NotImplementedError()
elif mode == 'test':
    # Model conversion to evaludation mode...
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    # Turn off back propagation...
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(dataloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                             test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    test_acc = 100. * correct / total

netscore = 20 * math.log10((test_acc ** 2) / (math.sqrt(param_mil) * math.sqrt(macs_bil)))

print("Test completed...")
print("NetScore : %.3f, Params(M) : %.3f, Mac(G) : %.3f, Test Acc : %.3f"
      %(netscore, param_mil, macs_bil, test_acc))

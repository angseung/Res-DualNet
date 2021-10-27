import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import torch.onnx

import os
import argparse
from utils import progress_bar, VisdomLinePlotter, VisdomImagePlotter, save_checkpoint
from models import *
from models.resnetCA import ResDaulNet18_TP5
from torchvision.models import mnasnet1_0, shufflenet_v2_x1_0, resnet18
from thop import profile

INPUT_SIZE = 32
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
# parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
max_epoch = 200



print(device)

# Data Preparing  !!!
print('==> Preparing data..')
# transform_train = transforms.Compose([
#     transforms.RandomCrop(32, padding=4),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.Resize((224,224))
# ])
#
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     transforms.Resize((224,224))
# ])

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                     std=[0.2023, 0.1994, 0.2010])

# normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
#                                  std = [0.229, 0.224, 0.225])

if INPUT_SIZE == 32:
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        normalize])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

elif INPUT_SIZE == 224:
    transform_train = transforms.Compose([
                transforms.RandomResizedCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])

    transform_test = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        normalize
    ])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
# trainset = torchvision.datasets.ImageNet(root='C:/imagenet/', split = 'train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testset = torchvision.datasets.ImageNet(root='C:/imagenet/', split = 'val', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=0)
# classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# bimage = VisdomImagePlotter('Train_batch')


# Training
def train(epoch, dir_path=None, plotter=None):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        # bimage.plot('input_batch', 't0', inputs)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        # bplot.plot('loss', 'train', 'Batch Loss', epoch * len(trainloader) + batch_idx, (train_loss / (batch_idx + 1)))
        # bplot.plot('acc', 'train', 'Batch Acc', epoch * len(trainloader) + batch_idx, 100. * correct / total)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    train_acc = 100. * correct / total
    plotter.plot('loss', 'train', 'Class Loss', epoch, train_loss / (batch_idx + 1))
    plotter.plot('acc', 'train', 'Class Accuracy', epoch, train_acc)

    with open('outputs/' + dir_path + '/log.txt', 'a') as f:
        f.write('Epoch [%d] |Train| Loss: %.3f, Acc: %.3f \t' % (
            epoch, train_loss / (batch_idx + 1), train_acc))

    return (epoch, train_acc, train_loss)


def test(epoch, dir_path=None, plotter=None):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    if dir_path is None:
        dir_path = 'outputs/checkpoint'
    else:
        dir_path = 'outputs/' + dir_path

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx,
                         len(testloader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (
                             test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    test_acc = 100. * correct / total

    # visualization
    plotter.plot('loss', 'val', 'Class Loss', epoch, test_loss / (batch_idx + 1))
    plotter.plot('acc', 'val', 'Class Accuracy', epoch, test_acc)

    # Save checkpoint.

    if test_acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        torch.save(state, './' + dir_path + '/ckpt.pth')
    #     torch.onnx.export(net,
    #                       torch.empty(1, 3, 224, 224, dtype=torch.float32, device=device),
    #                       dir_path + '/output.onnx')

        best_acc = test_acc

    with open(dir_path + '/log.txt', 'a') as f:
        f.write('|Test| Loss: %.3f, Acc: %.3f \n' % (test_loss / (batch_idx + 1), test_acc))

    return (epoch, test_acc, test_loss)


# Model
print('==> Building model..')

nets = {
    # 'mnasnet1_0_%s' %str(INPUT_SIZE) : mnasnet1_0(num_classes=10)
    # 'resnet18_%s' %str(INPUT_SIZE) : torchvision.models.resnet18(num_classes=10),
    'resdualnetSF_%s' % str(INPUT_SIZE): ResDaulNet18_TP5(),
    # 'shufflenet_v2_x1_0_%s' %str(INPUT_SIZE) : shufflenet_v2_x1_0(num_classes=10),
    # 'efficientnet_b0_%s' %str(INPUT_SIZE) : EfficientNetB0(num_classes=10),
    # 'mobilenet_%s' %str(INPUT_SIZE) : MobileNet(num_classes=10),
    # 'resnet18_%s' %str(INPUT_SIZE) : resnet18(num_classes=10),
    # 'clnet_v0_image_adam': CLNet_V0(1000),
        }

SAVE_CHECKPOINT = True

for netkey in nets.keys():
    # visualization
    plotter = VisdomLinePlotter(env_name='{} Training Plots'.format(netkey))
    log_path = 'outputs/' + netkey
    net = nets[netkey]
    net = net.to(device)

    from torchinfo import summary
    inputs = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
    os.makedirs(log_path, exist_ok=True)
    # flops, params = profile(net, inputs=(inputs, ))

    with open(log_path + '/log.txt', 'w') as f:
        f.write('Networks : %s\n' % netkey)
        m_info = str(summary(net, (1, 3, INPUT_SIZE, INPUT_SIZE), verbose=0))
        f.write('%s\n\n' % m_info)

    if device == 'cuda':
        net = torch.nn.DataParallel(net)  # Not support ONNX converting
        cudnn.benchmark = True

    # if args.resume:
    #     # Load checkpoint.
    #     print('==> Resuming from checkpoint..')
    #     assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #     checkpoint = torch.load('./checkpoint/ckpt.pth')
    #     net.load_state_dict(checkpoint['net'])
    #     best_acc = checkpoint['acc']
    #     start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=5e-4)
    # optimizer = optim.Adam(net.parameters(), lr=2e-4, betas=(0.5, 0.999), weight_decay=5e-4) ## Conf.2
    optimizer = optim.Adam(net.parameters(), lr=0.001)  ## Conf.2
    # optimizer = optim.RMSprop(net.parameters(), lr=0.256, alpha=0.99, eps=1e-08, weight_decay=0.9, momentum=0.9, centered=False) # Conf.1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(max_epoch * 1.0))
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.97, last_epoch=-1, verbose=True)
    # from lr_scheduler import CosineAnnealingWarmUpRestarts
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=2, eta_max=0.1,  T_up=10, gamma=0.5)

    for epoch in range(start_epoch, start_epoch + max_epoch):
        ep, train_acc, train_loss = train(epoch, netkey, plotter)
        ep, test_acc, test_loss = test(epoch, netkey, plotter)
        scheduler.step()

        if SAVE_CHECKPOINT:
            filename = log_path + '\%d_epoch.pth' %ep
            save_checkpoint(ep,
                            [train_acc, test_acc],
                            [train_loss, test_loss],
                            net,
                            optimizer,
                            filename)

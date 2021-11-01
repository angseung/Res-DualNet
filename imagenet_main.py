import torch
import torchinfo
import torch.backends.cudnn as cudnn
from models.resnetCA_ori import ResDaulNet18_TPI5

# Check use GPU or not
use_gpu = torch.cuda.is_available()  # use GPU

if use_gpu:
    device = torch.device("cuda:0")
else:
    raise NotImplementedError('CUDA Device needed to run this code...')

net = ResDaulNet18_TPI5()
net.to('cuda')
net = torch.nn.DataParallel(net)
cudnn.benchmark = True

# Load checkpoint data
# {net : net.state_dict,
#  acc : best test acc,
#  epoch : best performed epoch}

pth_path = "./outputs/resdual5_imagenet/ckpt.pth"
SAVEDAT = torch.load(pth_path)

net.load_state_dict(SAVEDAT['net'])
acc = SAVEDAT['acc']
epoch = SAVEDAT['epoch']

model_name = net.module.__class__.__name__
print("%s model was loaded successfully... [epoch : %03d] [best validation acc : %.3f]"
      %(model_name, epoch, acc))


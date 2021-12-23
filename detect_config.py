from models.detect_net import WpodNet
import torch
image_root = 'E:\python\pyProject\deepLearning\data\CCPD2019\ccpd_base'
batch_size = 32
weight = 'weights/wpod_net.pt'
epoch = 100000
net = WpodNet
device = 'cuda:0'
confidence_threshold = 0.9


device = torch.device(device if torch.cuda.is_available() else 'cpu')




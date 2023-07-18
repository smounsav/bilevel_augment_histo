from torch import nn
from .optimizers import get_optimizer
import torch 
from src import models
from torch.nn import functional as F

from src import utils as ut
import torch.optim as optim


from .base_netC import resnet_meta, resnet_meta_2, resnet_meta_old
import torchvision.models as models

from torchmeta.modules import MetaLinear
from torchmeta.modules import DataParallel

class Classifier(nn.Module):
    def __init__(self, model_dict, dataset, device):
        super().__init__()
        
        self.dataset = dataset

        self.model_dict = model_dict
        if self.model_dict['name'] == 'resnet18':
            if self.model_dict['pretrained']:
                self.net = models.resnet18(pretrained=True)
                self.net.fc = nn.Linear(512, self.dataset.n_classes)
            else:
                self.net = models.resnet18(num_classes= self.dataset.n_classes)
                
        elif self.model_dict['name'] == 'resnet18_meta':
            if self.model_dict.get('pretrained', True):
                self.net = resnet_meta.resnet18(pretrained=True)
                num_ftrs = self.net.fc.in_features
                self.net.fc = MetaLinear(num_ftrs, self.dataset.n_classes)
            else:
                self.net = resnet_meta.resnet18(num_classes= self.dataset.n_classes)
        elif self.model_dict['name'] == 'resnet18_meta_2':
                self.net = resnet_meta_2.ResNet18(nc=3, nclasses= self.dataset.n_classes)                

        elif self.model_dict['name'] == 'resnet18_meta_old':
                self.net = resnet_meta_old.ResNet18(nc=3, nclasses= self.dataset.n_classes)

        else:
            raise ValueError('network %s does not exist' % model_dict['name'])

        if (device.type == 'cuda'):
            self.net = DataParallel(self.net)
        self.net.to(device)
        # set optimizer
        self.opt_dict = model_dict['opt']
        self.lr_init = self.opt_dict['lr']
        self.opt = optim.SGD(self.net.parameters(), 
                                lr=self.opt_dict['lr'], 
                                momentum=self.opt_dict['momentum'], 
                                weight_decay=self.opt_dict['weight_decay'])

        # variables
        self.device = device

    def get_state_dict(self):
        state_dict = {'net': self.net.state_dict(),
                      'opt': self.opt.state_dict(),
                      }

        return state_dict

    def load_state_dict(self, state_dict):
        self.net.load_state_dict(state_dict['net'])
        self.opt.load_state_dict(state_dict['opt'])

    def on_trainloader_start(self, epoch):
        if self.opt_dict['sched']:
            ut.adjust_learning_rate_netC(self.opt, epoch, self.lr_init, self.model_dict['name'], self.dataset.name)

    def train_on_batch(self, batch):
        images, labels = batch['images'].to(self.device), batch['labels'].to(self.device) 
        logits = self.net(images)
        _, preds = torch.max(logits, 1)
        n_corr = (preds == labels).sum().item()        
        loss = F.cross_entropy(logits, labels, reduction="mean")

        self.opt.zero_grad()
        loss.backward()  

        self.opt.step()

        return loss.item(), n_corr
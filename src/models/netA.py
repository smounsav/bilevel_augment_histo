from torch import nn
from .optimizers import get_optimizer
import torch

import torch.optim as optim
from .base_netA import stn, small_affine, affine_color, color

from torch.nn import functional as F
from  src import utils as ut

from torchmeta.modules import DataParallel

class Augmenter(nn.Module):
    def __init__(self, model_dict, dataset, device):
        super().__init__()

        if model_dict['name'] == 'stn':
            self.net = stn.STN(isize=dataset.image_size,
                                    n_channels=dataset.nc, 
                                    n_filters=64, 
                                    nz=100, 
                                    datasetmean=dataset.mean, 
                                    datasetstd=dataset.std)

        elif model_dict['name'] == 'small_affine':
            self.net = small_affine.smallAffine(nz=6, 
                                            transformation=model_dict['transform'], 
                                            datasetmean=dataset.mean, 
                                            datasetstd=dataset.std)

        elif model_dict['name'] == 'affine_color':
            self.net = affine_color.affineColor(nz=100, 
                                            datasetmean=dataset.mean, 
                                            datasetstd=dataset.std)

        elif model_dict['name'] == 'color':
            self.net = color.color(nz=100, 
                                datasetmean=dataset.mean, 
                                datasetstd=dataset.std)

        else:
            raise ValueError('network %s does not exist' % model_dict['name'])

        if (device.type == 'cuda'):
            self.net = DataParallel(self.net)

        self.net.to(device)

        self.device = device
        self.factor = model_dict['factor']
        self.name = model_dict['name']
         
        if model_dict['name'] != 'random_augmenter':
            self.opt_dict = model_dict['opt']   
            self.lr_init = self.opt_dict['lr']            
            self.opt = optim.SGD(self.net.parameters(), 
                                 lr=self.opt_dict['lr'], 
                                 momentum=self.opt_dict['momentum'], 
                                 weight_decay=self.opt_dict['weight_decay'])

    def cycle(self, iterable):
        iterator = iter(iterable)
        while True:
            try:
                yield next(iterator)
            except StopIteration:
                iterator = iter(iterable)

    def get_state_dict(self):
        state_dict = {}
        if hasattr(self, 'opt'):
            state_dict['net'] =  self.net.state_dict()
            state_dict['opt'] = self.opt.state_dict()

        return state_dict

    def load_state_dict(self, state_dict):
        if hasattr(self, 'opt'):
            self.net.load_state_dict(state_dict['net'])
            self.opt.load_state_dict(state_dict['opt'])
    
    def apply_augmentation(self, images, labels):
        # apply augmentation to the given images
        factor = self.factor
        if factor > 1:
            labels = labels.repeat(factor)  
            images = images.repeat(factor, 1, 1, 1)

        with torch.autograd.set_detect_anomaly(True):
            augimages, transformations = self.net(images)

        return augimages, labels, transformations

    def on_trainloader_start(self, epoch, valloader, netC):
        # Update optimizer
        if self.opt_dict['sched']:
            ut.adjust_learning_rate_netA(self.opt, epoch, self.lr_init)

        # initialize momentums
        if netC.opt.defaults['momentum']:
            self.moms = {}
            for (name, p) in netC.net.named_parameters():
                self.moms[name] = torch.zeros(p.shape).to(self.device)

        self.epoch = epoch
        # Cycle through val_loader
        self.val_gen = self.cycle(valloader)

    def train_on_batch(self, batch, netC):
        self.train()
        images, labels = batch['images'].to(self.device), batch['labels'].to(self.device)      
        images, labels, transformations = self.apply_augmentation(images, labels)

        # Use classifier 
        logits = netC.net(images)
        _, preds = torch.max(logits, 1)
        n_corr = (preds == labels).sum().item()        
        loss_clf = F.cross_entropy(logits, labels, reduction="mean")
        
        netC.opt.zero_grad()

        if self.name in ['random_augmenter']:
            # Update the classifier only 
            loss_clf.backward() 
            netC.opt.step()

            return loss_clf

        elif self.name in ['stn']:
            # Update the style transformer network 
            self.opt.zero_grad()
            loss_clf.backward() 
            netC.opt.step()
            self.opt.step()

            return loss_clf
        
        else:
            # Update the augmenter through a validation batch
            # Calculate new weights w^t+1 to calculate the validation loss
            batch_val = next(self.val_gen)
            valimages, vallabels = batch_val['images'].to(self.device), batch_val['labels'].to(self.device)

            # construct graph
            loss_clf.backward(retain_graph=True)  
         
            self.w_t_1 = {}
            lr = ut.adjust_learning_rate_netC(netC.opt, self.epoch, netC.lr_init, netC.model_dict['name'], netC.dataset.name, return_lr=True) # get step size
 
            for (name, p) in netC.net.named_parameters():
                p.requires_grad = False # freeze C
                p.grad.requires_grad = True
                if netC.opt.defaults['momentum']:
                        self.moms[name] = netC.opt.defaults['momentum'] * self.moms[name] + p.grad # update momentums
                        self.w_t_1[name] = p - lr * self.moms[name] # compute future weights
                else:
                        self.w_t_1[name] = p - lr * p.grad # compute future weights

            # Calculate validation loss
            valoutput = netC.net(valimages, params=self.w_t_1)
            loss_aug = F.cross_entropy(valoutput, vallabels, reduction='mean')
            self.opt.zero_grad()
            loss_aug.backward()  
            self.opt.step()
            
            netC.opt.step()
            # After gradient is computed for A, unfreeze C
            for (name, p) in netC.net.named_parameters():
                p.requires_grad = True
                self.moms[name].detach_()

            self.w_t_1 = None

            return float(loss_clf.item()), n_corr, transformations

    def __call__(self, img):
        img = img.unsqueeze(0)
        img, _ = self.net.forward(img)
        img = img.squeeze(0)
        return img



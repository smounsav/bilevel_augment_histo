from collections import OrderedDict
import math

def adjust_learning_rate_netC(optimizer, epoch, lr_init, model, dataset, return_lr=False):

    if model in ['resnet18_meta', 'resnet18_meta_2']:
        lr = lr_init
    if return_lr:
            return lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def adjust_learning_rate_netA(optimizer, epoch, lr_init, return_lr=False):
    lr = lr_init
    if return_lr:
        return lr
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
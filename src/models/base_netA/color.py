import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform
from .color_utils import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue
# from kornia.enhance import adjust_brightness, adjust_contrast, adjust_saturation, adjust_hue

from math import pi

class color(nn.Module):
    def __init__(self, nz, datasetmean, datasetstd, neurons=6):
        super(color, self).__init__()
        self.nz = nz        
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)
        self.lin1 = nn.Linear(self.nz, 64)
        self.lin2 = nn.Linear(64, 32)
        self.lin3 = nn.Linear(32, 4)
        self.drop = nn.Dropout(0.2)

    def get_color_parameters(self, noise):
        colorparams = F.relu(self.lin1(noise))
        colorparams = self.drop(colorparams)
        colorparams = F.relu(self.lin2(colorparams))
        colorparams = self.drop(colorparams)
        colorparams = self.lin3(colorparams)
        colorparams = torch.tanh(colorparams)
        return colorparams

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)        
        # noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise = self.uniform.rsample()
        # get transformation parameters
        colorparams = self.get_color_parameters(noise)
        nb_transform = 4
        transform_order = torch.randperm(nb_transform).to(x.device)
        # Bring images back to [0:1]
        x = (x * self.std.view(1, 3, 1, 1)) + self.mean.view(1, 3, 1, 1)
        # apply transformation
        for i in range(nb_transform):
            if transform_order[i] == 0:
                x = adjust_brightness(x, 1 + colorparams[:, 0].squeeze(-1))
            elif transform_order[i] == 1:
                x = adjust_contrast(x, 1+ colorparams[:, 1].squeeze(-1))
            elif transform_order[i] == 2:
                x = adjust_saturation(x, 1 + colorparams[:, 2].squeeze(-1))
            elif transform_order[i] == 3:
                x = adjust_hue(x, colorparams[:, 3].squeeze(-1) * 0.5)
        # Restandardize images
        x = (x - self.mean.view(1, 3, 1, 1)) / self.std.view(1, 3, 1, 1)

        transformations = colorparams.detach().clone()

        return x, transformations
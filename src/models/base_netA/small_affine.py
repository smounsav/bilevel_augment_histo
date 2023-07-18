
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Uniform


class smallAffine(nn.Module):
    def __init__(self, nz, transformation, datasetmean, datasetstd, neurons=6):
        super(smallAffine, self).__init__()
        self.transformation = transformation
        self.nz = nz
        self.mean = torch.tensor(datasetmean)
        self.std = torch.tensor(datasetstd)

        self.lin1 = nn.Linear(self.nz, 64)
        self.lin2 = nn.Linear(64, 32)
        self.drop = nn.Dropout(0.2)

        if self.transformation == 'translation':
            self.lin3 = nn.Linear(32, 2)
        elif self.transformation == 'scale':
            self.lin3 = nn.Linear(32, 2)
        elif self.transformation == 'rotation':
            self.lin3 = nn.Linear(32, 4)
        elif self.transformation == 'affine':
            self.lin3 = nn.Linear(32, 6)

    # Regressor for the 3 * 2 affine matrix
    def get_affine_parameters(self, noise):
        affineparams = F.relu(self.lin1(noise))
        affineparams = self.drop(affineparams)
        affineparams = F.relu(self.lin2(affineparams))
        affineparams = self.drop(affineparams)
        return affineparams


    def get_affine_matrix(self, noise):
        identitymatrix = torch.eye(2, 3).to(noise.device)
        identitymatrix = identitymatrix.unsqueeze(0)
        identitymatrix = identitymatrix.repeat(noise.shape[0], 1, 1)
        theta = self.get_affine_parameters(noise)
        theta = self.lin3(theta)        
        if self.transformation  == 'translation':
            theta = torch.tanh(theta)
            affinematrix = identitymatrix
            affinematrix[:, :, 2] = theta
        elif self.transformation == 'scale':
            affinematrix = identitymatrix
            affinematrix[:, 0, 0] = theta[:, 0]
            affinematrix[:, 1, 1] = theta[:, 1]
        elif self.transformation == 'rotation':
            affinematrix = identitymatrix
            affinematrix[:, 0, 0] = theta[:, 0]
            affinematrix[:, 0, 1] = theta[:, 1]
            affinematrix[:, 1, 0] = theta[:, 2]
            affinematrix[:, 1, 1] = theta[:, 3]
        elif self.transformation == 'affine':
            theta = theta.view(-1, 2, 3)
            affinematrix = theta + identitymatrix

        return affinematrix

    def forward(self, x):
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        # noise
        bs = x.shape[0]
        self.uniform = Uniform(low=-torch.ones(bs, self.nz).to(x.device), high=torch.ones(bs, self.nz).to(x.device))
        noise = self.uniform.rsample()
        # get transformation matrix
        affinematrix = self.get_affine_matrix(noise)
        # compute transformation grid
        grid = F.affine_grid(affinematrix, x.size(), align_corners=True)
        # Bring back images to [-1;1]
        x = (x * self.std.view(1, 3, 1, 1))
        # apply transformation
        x = F.grid_sample(x, grid, align_corners=True)
        # Restandardize image
        x = x / self.std.view(1, 3, 1, 1)

        transformations = torch.mean(affinematrix.detach().clone().view(-1, 6), dim=0, keepdim=True)

        return x, transformations
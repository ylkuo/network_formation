import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable


class GMMRecognition(nn.Module):

    def __init__(self,RecognitionParams,xDim,yDim, nCUnits=100):
        '''
        h = Q_phi(x|y), where phi are parameters, x is our latent class, and y are data
        '''
        super(GMMRecognition, self).__init__()
        self.xDim=xDim
        self.yDim=yDim
        self.lin1 = nn.Linear(yDim, nCUnits)
        self.lin2 = nn.Linear(nCUnits, xDim)
        self.nonlinearity1 = nn.LeakyReLU(0.1)
        self.nonlinearity2 = nn.Softmax()

    def forward(self, x):
        y1 = self.nonlinearity1(self.lin1(x.float()))
        y2 = self.nonlinearity2(self.lin2(y1))
        return y2

    def getSample(self, Y):
        self.h = self.forward(Y)
        pi = np.asarray(torch.clamp(self.h, 0.001, 0.999).data)
        pi = (1/pi.sum(axis=1))[:, np.newaxis]*pi #enforce normalization (undesirable; for numerical stability)
        x_vals = np.zeros([pi.shape[0], self.xDim])
        for ii in range(pi.shape[0]):
            x_vals[ii,:] = np.random.multinomial(1, pi[ii], size=1)

        return Variable(torch.FloatTensor(x_vals.astype(bool) * 1.0))

    def evalLogDensity(self, hsamp, Y):

        ''' We assume each sample is a single multinomial sample from the latent h, so each sample is an integer class.'''
        self.h = self.forward(Y)
        return torch.log(torch.sum(self.h*hsamp, 1))


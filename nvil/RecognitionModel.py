import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.autograd import Variable

class RecognitionModel(object):
    '''
    Recognition Model Interace Class

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor must take the Input Theano variable and create the
    appropriate sampling expression.
    '''

    def __init__(self,xDim=3,yDim=2):
        self.xDim = xDim
        self.yDim = yDim

    def getSample(self):
        '''
        Returns a Theano object that are samples from the recognition model
        given the input
        '''
        raise Exception("Please implement me. This is an abstract method.")


class NeuralNetworkModel(nn.Module):
    def __init__(self, yDim=2, xDim=3, nCUnits=100):
        super(NeuralNetworkModel, self).__init__()
        self.lin1 = nn.Linear(yDim, nCUnits)
        self.lin2 = nn.Linear(nCUnits, xDim)
        self.nonlinearity1 = nn.LeakyReLU(0.1)
        self.nonlinearity2 = nn.Softmax()

    def forward(self, x):
        y1 = self.nonlinearity1(self.lin1(x.float()))
        y2 = self.nonlinearity2(self.lin2(y1))
        return y2

class GMMRecognition(RecognitionModel):

    def __init__(self,RecognitionParams,xDim,yDim):
        '''
        h = Q_phi(x|y), where phi are parameters, x is our latent class, and y are data
        '''
        super(GMMRecognition, self).__init__(xDim=xDim, yDim=yDim)
        self.network = NeuralNetworkModel(yDim)

    def getSample(self, Y):
        self.h = self.network.forward(Y)
        pi = np.asarray(torch.clamp(self.h, 0.001, 0.999).data)
        pi = (1/pi.sum(axis=1))[:, np.newaxis]*pi #enforce normalization (undesirable; for numerical stability)
        x_vals = np.zeros([pi.shape[0], self.xDim])
        for ii in range(pi.shape[0]):
            x_vals[ii,:] = np.random.multinomial(1, pi[ii], size=1)

        return x_vals.astype(bool)

    def evalLogDensity(self, hsamp, Y):

        ''' We assume each sample is a single multinomial sample from the latent h, so each sample is an integer class.'''
        self.h = np.asarray(self.network.forward(Y).data)
        return Variable(torch.FloatTensor(np.log((self.h*hsamp).sum(axis=1))))

    def parameters(self):
        return self.network.parameters()


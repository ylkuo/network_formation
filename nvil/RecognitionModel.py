import sys
sys.path.append('../lib/')

import torch


import torch.nn as nn
import torch.nn.functional as F

# import theano
# import lasagne
# from theano import tensor as T, function, printing
# import theano.tensor.nlinalg as Tla
import numpy as np
# import theano.tensor.slinalg as Tsla

class RecognitionModel(object):
    '''
    Recognition Model Interace Class

    Recognition model approximates the posterior given some observations

    Different forms of recognition models will have this interface

    The constructor must take the Input Theano variable and create the
    appropriate sampling expression.
    '''

    def __init__(self,indices,xDim,yDim):

        self.xDim = xDim
        self.yDim = yDim
        self.indices = indices

    # def evalEntropy(self):
    #     '''
    #     Evaluates entropy of posterior approximation
    #
    #     H(q(x))
    #
    #     This is NOT normalized by the number of samples
    #     '''
    #     raise Exception("Please implement me. This is an abstract method.")

    # def getParams(self):
    #     '''
    #     Returns a list of Theano objects that are parameters of the
    #     recognition model. These will be updated during learning
    #     '''
    #     return self.params

    def getSample(self):
        '''
        Returns a Theano object that are samples from the recognition model
        given the input
        '''
        raise Exception("Please implement me. This is an abstract method.")

    # def setTrainingMode(self):
    #     '''
    #     changes the internal state so that `getSample` will possibly return
    #     noisy samples for better generalization
    #     '''
    #     raise Exception("Please implement me. This is an abstract method.")
    #
    # def setTestMode(self):
    #     '''
    #     changes the internal state so that `getSample` will supress noise
    #     (e.g., dropout) for prediction
    #     '''
    #     raise Exception("Please implement me. This is an abstract method.")


class NeuralNetworkModel(nn.Module):

    # rec_nn = lasagne.layers.DenseLayer(rec_nn, 100, nonlinearity=leaky_rectify, W=lasagne.init.Orthogonal())
    # rec_nn = lasagne.layers.DenseLayer(rec_nn, xDim, nonlinearity=softmax, W=lasagne.init.Orthogonal(),
    #                                    b=-5 * np.ones(xDim, dtype=theano.config.floatX))
    #
    # NN_Params = dict([('network', rec_nn)])
    # recDict = dict([('NN_Params', NN_Params)])
    # self.NN_h = RecognitionParams['NN_Params']['network']

    def __init__(self, yDim=2, nCUnits=100):
        super(BiasCorrectNet, self).__init__()
        self.lin1 = nn.Linear(yDim, nCUnits)
        self.lin2 = nn.Linear(nCUnits, 1)
        self.nonlinearity1 = nn.LeakyReLU(0.1)
        self.nonlinearity2 = nn.Softmax()

    def forward(self, x):
        y1 = self.nonlinearity1(self.lin1(x))
        y2 = self.nonlinearity2(self.lin2(y1))
        return y2

class GMMRecognition(RecognitionModel,NeuralNetworkModel):

    def __init__(self,RecognitionParams,xDim,yDim):
        '''
        h = Q_phi(x|y), where phi are parameters, x is our latent class, and y are data
        '''
        super(GMMRecognition, self).__init__(indices=None,xDim=xDim,yDim=yDim)
        # self.N = Input.shape[0]                not used anywhere!

    # def getParams(self):
    #     network_params = lasagne.layers.get_all_params(self.NN_h)
    #     return network_params

    def getSample(self, Y):
        pi=T.clip(self.h, 0.001, 0.999).eval({self.Input:Y})
        pi= (1/pi.sum(axis=1))[:, np.newaxis]*pi #enforce normalization (undesirable; for numerical stability)
        x_vals = np.zeros([pi.shape[0],self.xDim])
        for ii in xrange(pi.shape[0]):
            x_vals[ii,:] = np.random.multinomial(1, pi[ii], size=1) #.nonzero()[1]

        return x_vals.astype(bool)

    def evalLogDensity(self, hsamp, Y):

        ''' We assume each sample is a single multinomial sample from the latent h, so each sample is an integer class.'''
        self.h = self.forward(Y)
        return torch.log((self.h*hsamp).sum(axis=1))

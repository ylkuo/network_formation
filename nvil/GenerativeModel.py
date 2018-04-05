import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable


def det(a):
    return torch.potrf(a).diag().prod()

def NormalPDFmat(X,Mu,XChol,xDim):
    ''' Use this version when X is a matrix [N x xDim] '''
    return torch.exp(logNormalPDFmat(X,Mu,XChol,xDim))

def logNormalPDFmat(X,Mu,XChol,xDim):
    ''' Use this version when X is a matrix [N x xDim] '''
    Lambda = torch.inverse(torch.mm(XChol,torch.t(XChol)))
    XMu = X-Mu
    return (-0.5 * torch.mm(XMu, torch.mm(Lambda, torch.t(XMu)))
            + 0.5 * X.shape[0] * torch.log(det(Lambda))
            - 0.5 * np.log(2*np.pi) * X.shape[0]*xDim)

def NormalPDF(X,Mu,XChol):
    return torch.exp(logNormalPDF(X,Mu,XChol))

def logNormalPDF(X,Mu,XChol):
    Lambda = torch.inverse(torch.mm(XChol,torch.t(XChol)))
    XMu    = X-Mu
    return (-0.5 * torch.mm(XMu.view(1,XMu.shape[0]), torch.mm(Lambda, XMu.view(XMu.shape[0], 1)))
            + 0.5 * torch.log(det(Lambda))
            - 0.5 * np.log(2*np.pi) * X.shape[0])


class GenerativeModel(object):
    '''
    Interface class for generative time-series models
    '''
    def __init__(self,GenerativeParams,xDim,yDim):

        # input variable referencing top-down or external input
        self.xDim = xDim
        self.yDim = yDim

    def evaluateLogDensity(self):
        '''
        Return a theano function that evaluates the density of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def getParams(self):
        '''
        Return parameters of the GenerativeModel.
        '''
        raise Exception('Cannot call function of interface class')

    def generateSamples(self):
        '''
        generates joint samples
        '''
        raise Exception('Cannot call function of interface class')

    def __repr__(self):
        return "GenerativeModel"


class MixtureOfGaussians(GenerativeModel):
    '''
    xDim - # classes
    yDim - dimensionality of observations
    '''
    def __init__(self, GenerativeParams, xDim, yDim):

        super(MixtureOfGaussians, self).__init__(GenerativeParams,xDim,yDim)

        # Mixture distribution
        if 'pi' in GenerativeParams:
            self.pi_un = nn.Parameter(torch.FloatTensor(np.asarray(GenerativeParams['pi'])), requires_grad=True)
        else:
            self.pi_un = nn.Parameter(torch.FloatTensor(np.asarray(100*np.ones(xDim))), requires_grad=True)
        self.pi = self.pi_un / self.pi_un.sum()

        if 'RChol' in GenerativeParams:
            self.RChol = nn.Parameter(torch.FloatTensor(np.asarray(GenerativeParams['RChol'])), requires_grad=True)
        else:
            self.RChol = nn.Parameter(torch.FloatTensor(np.asarray(np.random.randn(xDim, yDim, yDim)/5)), requires_grad=True)

        if 'mu' in GenerativeParams:
            self.mu = nn.Parameter(torch.FloatTensor(np.asarray(GenerativeParams['mu'])), requires_grad=True) # set to zero for stationary distribution
        else:
            self.mu = nn.Parameter(torch.FloatTensor(np.asarray(np.random.randn(xDim, yDim))), requires_grad=True)    # set to zero for stationary distribution


    def sampleXY(self, _N):
        _mu = np.asarray(self.mu.data)
        _RChol = np.asarray(self.RChol.data)
        _pi = np.asarray(torch.clamp(self.pi, 0.001, 0.999).data)

        b_vals = np.random.multinomial(1, _pi, size=_N)
        x_vals = b_vals.nonzero()[1]

        y_vals = np.zeros([_N, self.yDim])
        for ii in range(_N):
            y_vals[ii] = np.dot(np.random.randn(1,self.yDim), _RChol[x_vals[ii],:,:].T) + _mu[x_vals[ii]]

        b_vals = np.asarray(b_vals)
        y_vals = np.asarray(y_vals)

        return [b_vals, y_vals]

    def parameters(self):
        params_list = [self.RChol] + [self.mu] + [self.pi_un]
        for params in params_list:
            yield params

    def update_pi(self):
        self.pi = self.pi_un / self.pi_un.sum()

    def evaluateLogDensity(self, h, Y):
        X = torch.t(h.nonzero())[1]
        log_density = []
        for count in range(Y.shape[0]):
            LogDensityVeci = logNormalPDF(Y[count], torch.squeeze(self.mu[X[count]]), torch.squeeze(self.RChol[X[count]]))
            log_density += [LogDensityVeci + torch.log(self.pi[X[count]])]
        return torch.squeeze(torch.stack(log_density))

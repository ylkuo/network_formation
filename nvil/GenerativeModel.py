import torch

# import theano
# import lasagne
# import theano.tensor as T
# import theano.tensor.nlinalg as Tla
# import theano.tensor.slinalg as Tsla

import numpy as np

# from theano.tensor.shared_randomstreams import RandomStreams

def NormalPDFmat(X,Mu,XChol,xDim):
    ''' Use this version when X is a matrix [N x xDim] '''
    return torch.exp(logNormalPDFmat(X,Mu,XChol,xDim))

def logNormalPDFmat(X,Mu,XChol,xDim):
    ''' Use this version when X is a matrix [N x xDim] '''
    Lambda = torhc.inverse(torch.mm(XChol,torch.t(XChol)))
    XMu = X-Mu
    return (-0.5 * torch.mm(XMu, torch.mm(Lambda, torch.t(XMu)))
                  + 0.5 * X.shape[0] * torch.log(Tla.det(Lambda))
                  - 0.5 * np.log(2*np.pi) * X.shape[0]*xDim)

def NormalPDF(X,Mu,XChol):
    return torch.exp(logNormalPDF(X,Mu,XChol))

def logNormalPDF(X,Mu,XChol):
    Lambda = torch.inverse(torch.mm(XChol,torch.t(XChol)))  # torch.mm does matrix multiplication
    XMu    = X-Mu
    return (-0.5 * torch.mm(XMu, torch.mm(Lambda,torch.t(XMu)))  # torch.t transposes
                  + 0.5 * torch.log(torch.det(Lambda))
                  - 0.5 * np.log(2*np.pi) * X.shape[0])


class GenerativeModel(object):
    '''
    Interface class for generative time-series models
    '''
    def __init__(self,GenerativeParams,xDim,yDim):

        # input variable referencing top-down or external input

        self.xDim = xDim
        self.yDim = yDim


        # internal RV for generating sample
        # self.Xsamp = T.matrix('Xsamp')

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
            self.pi_un = torch.FloatTensor(np.asarray(GenerativeParams['pi']))
            self.pi_un_array = np.asarray(GenerativeParams['pi'])
            # theano.shared(value=np.asarray(GenerativeParams['pi'], dtype = theano.config.floatX), name='pi_un', borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.pi_un = torch.FloatTensor(np.asarray(100*np.ones(xDim)))
            self.pi_un_array = np.asarray(100 * np.ones(xDim))
            # theano.shared(value=np.asarray(100*np.ones(xDim), dtype = theano.config.floatX), name='pi_un' ,borrow=True)     # cholesky of observation noise cov matrix

        self.pi = torch.FloatTensor(self.pi_un_array/(self.pi_un_array).sum())

        if 'RChol' in GenerativeParams:
            self.RChol = torch.FloatTensor(np.asarray(GenerativeParams['RChol']), requires_grad=True)
            # theano.shared(value=np.asarray(GenerativeParams['RChol'], dtype = theano.config.floatX), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix
        else:
            self.RChol = torch.FloatTensor(np.asarray(np.random.randn(xDim, yDim, yDim)/5), requires_grad=True)
            # theano.shared(value=np.asarray(np.random.randn(xDim, yDim, yDim)/5, dtype = theano.config.floatX), name='RChol' ,borrow=True)     # cholesky of observation noise cov matrix

        if 'x0' in GenerativeParams:
            self.mu = torch.FloatTensor(np.asarray(GenerativeParams['mu']), requires_grad=True) # theano.shared(value=np.asarray(GenerativeParams['mu'], dtype = theano.config.floatX), name='mu',borrow=True)     # set to zero for stationary distribution
        else:
            self.mu = torch.FloatTensor(np.asarray(np.random.randn(xDim, yDim)), requires_grad=True) # theano.shared(value=np.asarray(np.random.randn(xDim, yDim), dtype = theano.config.floatX), name='mu', borrow=True)     # set to zero for stationary distribution


    def sampleXY(self,_N):

        _mu = self.mu #np.asarray(self.mu.data(), dtype=torch.FloatTensor)
        _RChol = np.asarray(self.RChol)
        _pi = torch.clamp(self.pi, 0.001, 0.999) #.data()
        # z = torch.clamp(x, 0, 1) will return a new Tensor with the result of x bounded between 0 and 1.

        b_vals = np.random.multinomial(1, _pi, size=_N)
        x_vals = b_vals.nonzero()[1]

        y_vals = np.zeros([_N, self.yDim])
        for ii in range(_N):
            y_vals[ii] = np.dot(np.random.randn(1,self.yDim), _RChol[x_vals[ii],:,:].T) + _mu[x_vals[ii]]

        b_vals = np.asarray(b_vals, dtype=torch.FloatTensor)
        y_vals = np.asarray(y_vals, dtype=torch.FloatTensor)

        return [b_vals, y_vals]

    def parameters(self):
        params_list = [self.RChol] + [self.mu] + [self.pi_un]
        for params in params_list:
            yield params
        # return [self.RChol] + [self.mu] + [self.pi_un]

    def evaluateLogDensity(self, h, Y):
        X = h.nonzero()[1]
        LogDensityVec = []
        for count in range(length(Y)):
            LogDensityVeci, _ = logNormalPDF(Y[count], self.mu[X][count], self.RChol[X][count])
            LogDensityVec += [LogDensityVeci]
        # LogDensityVec,_ = torch.map(logNormalPDF, sequences = [Y,self.mu[X],self.RChol[X]])
        return LogDensityVec + torch.log(self.pi[X])

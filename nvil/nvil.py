# Class for Neural Variational Inference and Learning (NVIL)

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.autograd import Variable

class BiasCorrectNet(nn.Module):
    def __init__(self, yDim=2, nCUnits=100):
        super(BiasCorrectNet, self).__init__()
        self.lin1 = nn.Linear(yDim, nCUnits)
        self.lin2 = nn.Linear(nCUnits, 1)
        self.nonlin = nn.LeakyReLU()
    def forward(self, x):
        x = self.nonlin(self.lin1(x))
        return F.relu(self.lin2(x))

class NVIL():
    def __init__(self, 
                 opt_params, # dictionary of optimization parameters
                 gen_params, # dictionary of generative model parameters
                 GEN_MODEL,  # class that inherits from GenerativeModel
                 rec_params, # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL, # class that inherits from RecognitionModel
                 xDim=2, # dimensionality of latent state
                 yDim=2, # dimensionality of observations
                 nCUnits=100, # number of units used in the (single-layer) bias-correction network
                 learning_rate=3e-4
                ):

        self.xDim   = xDim
        self.yDim   = yDim
        
        #---------------------------------------------------------
        # instantiate our prior & recognition models
        self.mrec   = REC_MODEL(rec_params, self.xDim, self.yDim)
        self.mprior = GEN_MODEL(gen_params, self.xDim, self.yDim)

        # NVIL Bias-correction network
        self.C_nn = BiasCorrectNet(yDim, nCUnits)
 
        # Set NVIL params
        self.c = torch.FloatTensor([opt_params['c0']])
        self.v = torch.FloatTensor([opt_params['v0']])
        self.alpha = torch.FloatTensor([opt_params['alpha']])

        # ADAM defaults
        self.mprior_opt = optim.Adam(self.mprior.parameters(),
                                     lr=learning_rate)
        self.mrec_opt = optim.Adam(self.mrec.parameters(),
                                   lr=learning_rate)
        self.C_nn_opt = optim.Adam(self.C_nn.parameters(),
                                   lr=learning_rate)

    def get_nvil_cost(self, Y, hsamp):
        # First, compute L and l (as defined in Algorithm 1 in Gregor & ..., 2014)

        # Evaluate the recognition model density Q_\phi(h_i | y_i)
        q_hgy = self.mrec.evalLogDensity(hsamp, Y)
        # Evaluate the generative model density P_\theta(y_i , h_i)
        p_yh =  self.mprior.evaluateLogDensity(hsamp, Y)
        C_out = torch.squeeze(self.C_nn.forward(Y))
        L = p_yh.mean() - q_hgy.mean()
        l = p_yh - q_hgy - C_out
        return [L, l, p_yh, q_hgy, C_out]

    def update_cv(self, l, batch_y, h):
        # Now compute derived quantities for the update
        cb = l.mean().data
        vb = l.var().data
        self.c = self.alpha * self.c + (1-self.alpha) * cb
        self.v = self.alpha * self.v + (1-self.alpha) * vb

    def update_params(self, y, l, p_yh, q_hgy, C_out):
        loss = 0
        for i in range(y.shape[0]):
            if torch.sqrt(self.v).numpy()[0] > 1.0:
                lii = (l[i] - Variable(self.c)) / Variable(torch.sqrt(self.v))
            else:
                lii = l[i] - Variable(self.c)
            cost = p_yh[i] + q_hgy[i] * lii + C_out[i] * lii
            loss += cost
        # compute gradients
        loss.backward()
        # step to optimize
        self.mprior_opt.zero_grad()
        self.mrec_opt.zero_grad()
        self.C_nn_opt.zero_grad()
        self.mprior_opt.step()
        self.mrec_opt.step()
        self.C_nn_opt.step()

    def fit(self, data_loader, max_epochs=100):
        avg_costs = []
        epoch = 0
        while epoch < max_epochs:
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            batch_counter = 0
            for data_x, data_y in data_loader:
                y = Variable(data_y).float()
                hsamp_np = self.mrec.getSample(y)
                L, l, p_yh, q_hgy, C_out = self.get_nvil_cost(y, hsamp_np)
                self.update_cv(l, y, hsamp_np)
                self.update_params(y, l, p_yh, q_hgy, C_out)
                if np.mod(batch_counter, 10) == 0:
                    cx = self.c
                    vx = self.v
                    print('(c, v, L): (%f, %f, %f)\n' % (np.asarray(cx), np.asarray(vx), L))
                avg_costs.append(L.data.numpy())
                batch_counter += 1
            epoch += 1
        return avg_costs

# An example to use NVIL with GMM data and model.

import numpy as np
import torch

from dataset import GMMDataset
from GenerativeModel import MixtureOfGaussians
from nvil import *
from RecognitionModel import GMMRecognition
from matplotlib import pyplot as plt
from plot_cov import *
from sklearn.cluster import KMeans
from torch.autograd import Variable
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # load dataset
    dataset = GMMDataset()
    data_loader = DataLoader(dataset=dataset,
                             batch_size=10, shuffle=True)

    # build model
    opt_params = dict({'c0': -0.0, 'v0': 1.0, 'alpha': 0.9})
    NN_Params = dict([])
    recDict = dict([('NN_Params', NN_Params)])
    xdim, ydim = dataset.get_dim()
    model = NVIL(opt_params, dict([]), MixtureOfGaussians,
                 recDict, GMMRecognition, xdim, ydim,
                 nCUnits=100, learning_rate=3e-4)

    # init generative model with k-means solution
    km = KMeans(n_clusters=xdim, n_init=10, max_iter=500)
    kmpred = km.fit_predict(dataset.ytrain)
    km_mu = np.zeros([xdim, ydim])
    km_chol = np.zeros([xdim, ydim, ydim])
    for cl in np.unique(kmpred):
        km_mu[cl] = dataset.ytrain[kmpred == cl].mean(axis=0)
        km_chol[cl] = np.linalg.cholesky(np.cov(dataset.ytrain[kmpred == cl].T))
    km_pi = np.histogram(kmpred, bins=xdim)[0] / (1.0*kmpred.shape[0])
    model.mprior.pi_un.data = torch.FloatTensor(km_pi)
    model.mprior.mu.data = torch.FloatTensor(km_mu)
    model.mprior.RChol.data = torch.FloatTensor(km_chol)

    # fit the model
    costs = model.fit(data_loader, max_epochs=5)

    # plot ELBO
    plt.figure()
    plt.plot(costs)
    plt.axis('tight')
    plt.xlabel('iteration')
    plt.ylabel('ELBO\n(averaged over minibatch)')
    plt.show()

    # plot learned distribution
    clr = ['b', 'r', 'c', 'g', 'm', 'o']
    plt.figure()
    plt.subplot(121)
    plt.plot(dataset.ysamp[:,0], dataset.ysamp[:,1], 'k.', alpha=.1)
    plt.hold('on')
    for ii in range(xdim):
        Rc = dataset.gmm.RChol[ii].data.numpy()
        plot_cov_ellipse(Rc.dot(Rc.T), dataset.gmm.mu[ii].data.numpy(), nstd=2, color=clr[ii], alpha=.3)
    plt.title('True Distribution')
    plt.ylabel(r'$x_0$')
    plt.xlabel(r'$x_1$')
    plt.subplot(122)
    plt.hold('on')
    plt.plot(dataset.ytrain[:,0], dataset.ytrain[:,1],'k.', alpha=.1)
    for ii in range(xdim):
        Rc = model.mprior.RChol[ii].data.numpy()
        plot_cov_ellipse(Rc.dot(Rc.T), model.mprior.mu[ii].data.numpy(), nstd=2, color=clr[ii], alpha=.3)
    plt.title('Learned Distributions')    
    plt.ylabel(r'$x_0$')
    plt.xlabel(r'$x_1$')
    plt.show()

    # plot inferred label
    xlbl = dataset.xsamp.nonzero()[1]
    post_dist = model.mrec.forward(Variable(torch.FloatTensor(dataset.ytrain))).data.numpy()
    learned_lbl = post_dist.argmax(axis=1)
    clr = ['b', 'r', 'c', 'g', 'm', 'o']
    plt.figure()
    for ii in np.random.permutation(range(500)):
        plt.subplot(121)
        plt.hold('on')
        plt.plot(dataset.ysamp[ii,0], dataset.ysamp[ii,1], '.', color=clr[xlbl[ii]])
        plt.subplot(122)
        plt.hold('on')
        plt.plot(dataset.ysamp[ii,0], dataset.ysamp[ii,1], '.', color=clr[learned_lbl[ii]])
    plt.subplot(121)
    plt.title('True Label')
    plt.ylabel(r'$x_0$')
    plt.xlabel(r'$x_1$')
    plt.subplot(122)
    plt.title('Inferred Label')
    plt.ylabel(r'$x_0$')
    plt.xlabel(r'$x_1$')
    plt.show()

# An example to use NVIL with GMM data and model.

import numpy as np

from dataset import GMMDataset
from GenerativeModel import MixtureOfGaussians
from nvil import *
from RecognitionModel import GMMRecognition
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader

def __main__():
    # load dataset
    data_loader = DataLoader(dataset=GMMDataset(),
                             batch_size=10, shuffle=True)

    # build model
    opt_params = dict({'c0': -0.0, 'v0': 1.0, 'alpha': 0.9})
    # TODO: call to get the recognition network
    NN_Params = dict([('network', rec_nn)])
    recDict = dict([('NN_Params', NN_Params)])
    xdim, ydim = data_loader.get_dim()
    model = NVIL(opt_params, dict([]), MixtureOfGaussians,
                 recDict, GMMRecognition, xdim, ydim,
                 nCUnits=100, learning_rate=3e-4)

    # init generative model with k-means solution
    km = KMeans(n_clusters=xdim, n_init=10, max_iter=500)
    kmpred = km.fit_predict(data_loader.ytrain)
    km_mu = np.zeros([xdim, ydim])
    km_chol = np.zeros([xdim, ydim, ydim])
    for cl in np.unique(kmpred):
        km_mu[cl] = ytrain[kmpred == cl].mean(axis=0)
        km_chol[cl] = np.linalg.cholesky(np.cov(ytrain[kmpred == cl].T))
    km_pi = np.histogram(kmpred, bins=xdim)[0] / (1.0*kmpred.shape[0])
    model.mprior.pi_un.set_value(km_pi)
    model.mprior.mu.set_value(km_mu)
    model.mprior.RChol.set_value(km_chol)

    # fit the model
    costs = model.fit(data_loader, max_epochs=5)

    # plot ELBO
    plt.figure()
    plt.plot(costs)
    plt.axis('tight')
    plt.xlabel('iteration')
    plt.ylabel('ELBO\n(averaged over minibatch)')

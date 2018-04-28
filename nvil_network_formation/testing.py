# Use NVIL to make inference about the structural homophily in the network formation model

from dataset import NetworkDataset, NetworkIterator
from GenerativeModel import NetworkFormationGenerativeModel
from nvil import *
from RecognitionModel import NetworkFormationRecognition
from matplotlib import pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader

import settings

if __name__ == '__main__':
    # load dataset
    dataset = NetworkDataset(N=10)

    # training_samples = Simulations_Dataset(n_iters, features, labels)
    # training_samples_loader = utils_data.DataLoader(training_samples, batch_size,collate_fn=PadCollate(dim=0))

    #data_loader = DataLoader(dataset=dataset,
    #                         batch_size=1, shuffle=True) # for batch_size>1 need to take care of padding time series
    # to equal lengths

    # build model
    opt_params = dict({'c0': -0.0, 'v0': 1.0, 'alpha': 0.9})
    # NN_Params = dict([])
    # rec_params = dict([('NN_Params', NN_Params)])
    xdim = dataset.get_dim()
    gen_params = dict([])
    if settings.load_model:
        model = NVIL.load_model(settings.load_model_path)
    else:
        model = NVIL(opt_params, settings.gen_model_params, NetworkFormationGenerativeModel,
                     NetworkFormationRecognition, xdim, learning_rate=3e-3)

    km_pi = list(np.linspace((settings.class_values[0]-1),(settings.class_values[-1]+1),100))
    print(km_pi)
    model.generative_model.prior.data = torch.FloatTensor(km_pi)
  
    # fit the model
    costs = model.fit(dataset, batch_size=20, max_epochs=1, save=True)

    # eval posterior estimators
    # TODO: plot theta vs estimate thetas figure
    # for theta in [2,3,4,5,6]:
    #     error, thetas = eval_posterior(theta, model.recognition_model, model.generative_model,
    #                                    n_samples=1, n_thetas=10,
    #                                    estimator_type='posterior_mean',
    #                                    which_posterior='variational')
    #     print('theta, estimate_thetas, error: %f, %r, %f' % (theta, thetas, error))

    true_thetas = [2,3,4,5,6]
    estimator = Estimator(true_thetas, model.recognition_model, model.generative_model, n_samples=5,
                          n_posterior_samples=10,
                          estimator_type='posterior_mean', bin_size=5, which_posterior='variational', error_type='MSE')
    estimator.get_estimates_for_true_thetas(true_thetas, do_plot=True, symmetric=False)

    # plot ELBO
    plt.figure()
    plt.plot(costs)
    plt.axis('tight')
    plt.xlabel('iteration')
    plt.ylabel('ELBO\n(averaged over minibatch)')
    plt.show()

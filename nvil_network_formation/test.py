# Use NVIL to make inference about the structural homophily in the network formation model

from dataset import NetworkFormationDataset
from GenerativeModel import NetworkFormationGenerativeModel
from nvil import *
from RecognitionModel import NetworkFormationRecognition
from matplotlib import pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # load dataset
    dataset = NetworkFormationDataset()

    # training_samples = Simulations_Dataset(n_iters, features, labels)
    # training_samples_loader = utils_data.DataLoader(training_samples, batch_size,collate_fn=PadCollate(dim=0))

    data_loader = DataLoader(dataset=dataset,
                             batch_size=1, shuffle=True) # for batch_size>1 need to take care of padding time series
    # to equal lengths

    # build model
    opt_params = dict({'c0': -0.0, 'v0': 1.0, 'alpha': 0.9})
    NN_Params = dict([])
    rec_params = dict([('NN_Params', NN_Params)])
    xdim = dataset.get_dim()
    gen_params = dict([])
    model = NVIL(opt_params, gen_params, NetworkFormationGenerativeModel,
                 rec_params, NetworkFormationRecognition, xdim, learning_rate=3e-4)

    km_pi = [1/3, 1/3 , 1/3]
    model.generative_model.pi_un.data = torch.FloatTensor(km_pi)
  
    # fit the model
    costs = model.fit(data_loader, max_epochs=5)

    # plot ELBO
    plt.figure()
    plt.plot(costs)
    plt.axis('tight')
    plt.xlabel('iteration')
    plt.ylabel('ELBO\n(averaged over minibatch)')
    plt.show()

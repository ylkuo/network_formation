import torch
import settings
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class recognition_RNN(nn.Module):
    def __init__(self, input_size = settings.number_of_features, hidden_size = settings.n_hidden,
                 num_layers = settings.NUM_LAYERS, output_size = settings.OUTPUT_SIZE,nCUnits = 100):
        super(recognition_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rec = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )#batch_first=False the first dimension is Time, the second dimension is Batch

        self.lin1 = nn.Linear(hidden_size, nCUnits)
        # self.lin1 = nn.Linear(yDim, nCUnits)
        self.nonlinearity1 = nn.LeakyReLU(0.1)
        self.lin2 = nn.Linear(nCUnits, output_size)
        self.nonlinearity2 = nn.Softmax()
        # self.training_losses = []

    def forward(self, input, hidden):
        output_r, _ = self.rec(input, hidden)  # the recurrent units output
        output_n = self.nonlin(output_r[-1])  # non-linearity after the recurrent units
        output_l = self.lin2(self.lin1(output_n))  # second linear layer
        output = self.nonlinearity2(output_l)  # final non-linearity is a softmax
        return output

    def initHidden(self):
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

    def getSamplePrediction(self,torchSample):
        hidden0 = self.initHidden()
        # print(torchSample)
        if isinstance(torchSample,list) or isinstance(torchSample,tuple):
            feature = torchSample[1]
        else:
            feature = torchSample
        output_label = self.__call__(Variable(feature), hidden0)
        _, prediction = torch.topk(output_label, 1)
        return prediction.data[0][0], np.exp(output_label.data[0][prediction.data[0][0]])

    def getSampleOutput(self,torchSample):
        hidden0 = self.initHidden()
        # print(torchSample)
        if isinstance(torchSample,list) or isinstance(torchSample,tuple):
            feature = torchSample[1]
        else:
            feature = torchSample
        output = self.__call__(Variable(feature), hidden0)

        return output



class NetworkFormationRecognition(recognition_RNN):

    def __init__(self, number_of_classes):
        '''
        h = Q_phi(x|y), where phi are parameters, x is our latent class, and y are data
        '''
        super(NetworkFormationRecognition, self).__init__()
        self.number_of_classes=number_of_classes

    def getSample(self, Y):
        # Y is generated by sampleXY in the generative model.
        # note that each sample is a dictionary with two keys: 'degrees' and 'network_time_series', only the degrees is
        # supplied as input to the recognition network
        hidden0 = self.initHidden()
        input_degrees = Y['degrees']
        self.h = self.__call__(Variable(input_degrees), hidden0)
        # self.h = self.forward(input_degrees)  # this is the (classification) neural network output,
        # posterior probabilities for each class
        pi = np.asarray(torch.clamp(self.h, 0.001, 0.999).data)
        pi = (1/pi.sum(axis=1))[:, np.newaxis]*pi #enforce normalization (undesirable; for numerical stability)
        x_vals = np.zeros([pi.shape[0], self.number_of_classes])
        for ii in range(pi.shape[0]):
            x_vals[ii,:] = np.random.multinomial(1, pi[ii], size=1)

        return Variable(torch.FloatTensor(x_vals.astype(bool) * 1.0))

    def evalLogDensity(self, hsamp, Y):

        ''' We assume each sample is a single multinomial sample from the latent h, so each sample is an integer class.'''
        self.h = self.forward(Y['degrees'])
        return torch.log(torch.sum(self.h*hsamp, 1))

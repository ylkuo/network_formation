# Class for Neural Variational Inference and Learning (NVIL)

import numpy as np

import settings

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class bias_correction_RNN(nn.Module):
    def __init__(self, input_size=settings.number_of_features, hidden_size=settings.n_hidden,
                 num_layers=settings.NUM_LAYERS, output_size=1):
        super(bias_correction_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rec = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=False,
        )  # batch_first=False the first dimension is Time, the second dimension is Batch
        self.nonlin = nn.LeakyReLU()
        self.lin1 = nn.Linear(hidden_size, output_size)
        self.lin2 = nn.Linear(output_size, output_size)

        # self.training_losses = []

    def forward(self, input, hidden):
        output_r, _ = self.rec(input, hidden)  # the recurrent units output
        output_r.view(-1, self.hidden_size)
        output_n = self.nonlin(output_r)  # non-linearity after the recurrent units
        # self.softmax(self.out(output[-1])) #F.log_softmax(self.out(output[-1]))
        output = self.lin2(self.lin1(output_n)) # final linear layers
        #print('output of bias correction RNN',output)
        return output

    def initHidden(self):
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size)))

    def getSampleOutput(self, torchSample):
        hidden0 = self.initHidden()
        # print(torchSample)
        if isinstance(torchSample, list) or isinstance(torchSample, tuple):
            feature = torchSample[1]
        else:
            feature = torchSample
        output = self.__call__(Variable(feature), hidden0)

        return output

class NVIL():
    def __init__(self, 
                 opt_params, # dictionary of optimization parameters
                 gen_params, # dictionary of generative model parameters
                 GEN_MODEL,  # class that inherits from GenerativeModel
                 # rec_params, # dictionary of approximate posterior ("recognition model") parameters
                 REC_MODEL, # class that inherits from RecognitionModel
                 number_of_classes=2, # dimensionality of latent state
                 learning_rate=3e-4):

        self.number_of_classes = number_of_classes

        #---------------------------------------------------------
        # instantiate our prior & recognition models
        self.recognition_model = REC_MODEL(self.number_of_classes)
        # print(gen_params)
        self.generative_model = GEN_MODEL(gen_params)

        # NVIL Bias-correction network
        self.bias_correction = bias_correction_RNN(input_size = settings.number_of_features, hidden_size = settings.n_hidden,
                 num_layers = settings.NUM_LAYERS, output_size = 1)
 
        # Set NVIL params
        self.c = torch.FloatTensor([opt_params['c0']])
        self.v = torch.FloatTensor([opt_params['v0']])
        self.alpha = torch.FloatTensor([opt_params['alpha']])

        # ADAM defaults
        self.opt = optim.Adam(list(self.generative_model.parameters()) + list(self.recognition_model.parameters()) +
                              list(self.bias_correction.parameters()),
                              lr=learning_rate)

    def get_nvil_cost(self, Y, hsamp):
        # First, compute L and l (as defined in Algorithm 1 in Gregor & ..., 2014)
        # Evaluate the recognition model density Q_\phi(h_i | y_i)
        # print('hsamp fed into recognition model eval log density',hsamp)
        # print('Y fed into recognition model eval log density', Y)
        q_hgy = self.recognition_model.evalLogDensity(hsamp, Y)

        # Evaluate the generative model density P_\theta(y_i , h_i)
        p_yh = self.generative_model.evaluateLogDensity(hsamp, Y)
        hidden0 = self.bias_correction.initHidden()
        input_degrees = Y['degrees'].unsqueeze(0)
        input_degrees = input_degrees.permute(1, 0, 2)
        #print('input_degrees in bias correction',input_degrees)
        C_out = torch.squeeze(self.bias_correction.__call__(Variable(input_degrees), hidden0))
        #print('C_out in bias correction',C_out)
        L = p_yh.mean() - q_hgy.mean()
        #print('q_hgy', q_hgy)
        #print('p_yh', p_yh)
        l = p_yh - q_hgy - C_out
        #print('l',l)
        return [L, l, p_yh, q_hgy, C_out]

    def update_cv(self, l):
        # Now compute derived quantities for the update
        cb = l.mean().data
        vb = l.var().data
        self.c = self.alpha * self.c + (1-self.alpha) * cb
        self.v = self.alpha * self.v + (1-self.alpha) * vb

    def update_params(self, batch_size, l, p_yh, q_hgy, C_out):
        self.opt.zero_grad()
        for i in range(batch_size):
            if torch.sqrt(self.v).numpy()[0] > 1.0:
                lii = (l[i].data - self.c) / torch.sqrt(self.v)
            else:
                lii = l[i].data - self.c
            lii = Variable(torch.FloatTensor(lii), requires_grad=False)
            loss_p = p_yh[i] * -1
            loss_p.backward(retain_graph=True)
            loss_q = q_hgy[i] * lii * -1
            loss_q.backward(retain_graph=True)
            loss_C = C_out[i] * lii * -1
            loss_C.backward(retain_graph=True)
        self.opt.step()
        self.generative_model.update_pi()

    def fit(self, data_loader, max_epochs=100):
        avg_costs = []
        epoch = 0
        while epoch < max_epochs:
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            batch_counter = 0
            for data_x, data_y in data_loader:
                #print('data_x', data_x)
                #print('data_y', data_y)
                hsamp_np = self.recognition_model.getSample(data_y)
                #print('hsamp_np',hsamp_np)
                L, l, p_yh, q_hgy, C_out = self.get_nvil_cost(data_y, hsamp_np)
                self.update_cv(l)
                #print('data_y fed into update_params',data_y)
                self.update_params(1, l, p_yh, q_hgy, C_out)# data_y.shape[0] for batch_size
                if np.mod(batch_counter, 10) == 0:
                    cx = self.c
                    vx = self.v
                    print('(c, v, L): (%f, %f, %f)\n' % (np.asarray(cx), np.asarray(vx), L))
                avg_costs.append(L.data.numpy())
                batch_counter += 1
            epoch += 1
        return avg_costs


 # def doTraining(self, dataset, n_iters  = SENTINEL , batch_size=4, window_length_loss=64 , verbose = False ,
    #                save = False , file_name = 'model.pkl'):
    #
    #     if n_iters is SENTINEL:
    #         n_iters = len(dataset)
    #
    #     features = []
    #     labels = []
    #     for iter in dataset:
    #         labels.append(iter[0])
    #         features.append(iter[1])
    #
    #     training_samples = Simulations_Dataset(n_iters, features, labels)
    #     training_samples_loader = utils_data.DataLoader(training_samples, batch_size,collate_fn=PadCollate(dim=0))#
    #
    #
    #     criterion = nn.NLLLoss()
    #     optimizer = torch.optim.Adam(self.parameters(), lr=0.002, weight_decay=0)
    #
    #     moving_avg_losses = []
    #     current_agg_loss = 0
    #
    #     count_iterations = 0
    #     start = time.time()
    #     total_number_of_rounds = n_iters // batch_size
    #
    #     for i, samples in enumerate(training_samples_loader):
    #         if i == total_number_of_rounds:
    #             break
    #         # Get each batch
    #         sampled_features, sampled_labels = samples  # features[i], labels[i] #samples
    #         # Convert tensors into Variables
    #         sampled_features, sampled_labels = Variable(sampled_features.permute(1, 0, 2)), Variable(sampled_labels)
    #
    #         hidden = self.initHidden()
    #         output_type = self.__call__(sampled_features, hidden)
    #
    #         optimizer.zero_grad()
    #         #print(output_type)
    #         #print('next')
    #         #print(sampled_labels)
    #         loss_type = criterion(output_type, sampled_labels.view(batch_size))
    #
    #         loss_type.backward()
    #
    #         optimizer.step()
    #
    #         loss = loss_type.data[0]
    #         count_iterations += 1
    #         current_agg_loss += loss
    #
    #         if count_iterations % window_length_loss == 0:
    #             current_avg_loss = current_agg_loss / window_length_loss
    #             moving_avg_losses.append(current_avg_loss)
    #             if verbose:
    #                 print('%d %d%% (%s) %.4f (avg %.4f) ' %
    #                     (count_iterations, float(count_iterations) / total_number_of_rounds * 100, timeSince(start), loss,
    #                      current_avg_loss))
    #
    #             current_agg_loss = 0
    #     self.training_losses = self.training_losses + moving_avg_losses
    #     if save:
    #         model_fine_tuned = copy.deepcopy(self)
    #         pickle.dump(model_fine_tuned, open( './data/'+ file_name, 'wb'))
    #     return moving_avg_losses
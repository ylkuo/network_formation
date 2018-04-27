# Class for Neural Variational Inference and Learning (NVIL)

import numpy as np

import pickle, settings

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from dataset import NetworkIterator

from matplotlib import pyplot as plt

# estimator_type can be 'posterior_mean', 'MAP', 'median'
# which_posterior can be 'exact', 'variational'
def eval_posterior(theta, rec_model, gen_model, n_samples=5, n_thetas=10,
                   estimator_type='MAP', bin_size=5, which_posterior='variational'):
    selected_thetas = np.zeros(n_samples)
    for i in range(n_samples):
        y_val = dict().fromkeys(('network', 'degrees'))
        y_val['network'], y_val['degrees'] = gen_model.get_y(theta)
        sampled_thetas = np.zeros(n_thetas)

        if which_posterior == 'variational':
            for j in range(n_thetas):
                sampled_theta = rec_model.getSample(y_val)
                sampled_thetas[j] = sampled_theta.data[0]
        elif which_posterior == 'exact':
            sampled_thetas = gen_model.get_exact_posterior_samples(y_val,n_thetas)
        else:
            assert False, "psoterior type is invalid."
        print('sampled_thetas:',sampled_thetas)
        plt.hist(sampled_thetas)
        plt.show()
        if estimator_type == 'posterior_mean':
            selected_thetas[i] = sampled_thetas.mean()
        elif estimator_type == 'MAP':
            hist, bin_edges = np.histogram(sampled_thetas, bins=bin_size)
            j = np.argmax(hist)
            selected_thetas[i] = (bin_edges[j] + bin_edges[j+1]) / 2.0
        elif estimator_type == 'median':
            selected_thetas[i] = np.median(sampled_thetas)
    error = 0
    if estimator_type == 'posterior_mean':
        error = np.sum((selected_theta - theta)**2 for selected_theta in selected_thetas)
    elif estimator_type == 'MAP' or 'median':
        error = np.sum(abs(selected_theta - theta) for selected_theta in selected_thetas)
    error /= n_samples
    return error, selected_thetas

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
        #print('output_r', output_r.shape)
        output_r = output_r.view(-1, self.hidden_size)
        output_n = self.nonlin(output_r)  # non-linearity after the recurrent units
        # self.softmax(self.out(output[-1])) #F.log_softmax(self.out(output[-1]))
        output = self.lin2(self.lin1(output_n)) # final linear layers
        #print('output of bias correction RNN',output)
        # print('self.lin2.bias:', self.lin2.bias)
        return output[-1]

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
                 learning_rate=3e-4,
                 use_load_model=False,
                 bias_model=None):

        self.number_of_classes = number_of_classes

        #---------------------------------------------------------
        if use_load_model:
            self.recognition_model = REC_MODEL
            self.generative_model = GEN_MODEL
            self.bias_correction = bias_model
            self.c = opt_params['c0']
            self.v = opt_params['v0']
            self.alpha = opt_params['alpha']
        else:
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
        self.learning_rate = learning_rate
        self.opt = optim.Adam(list(self.generative_model.parameters()) + list(self.recognition_model.parameters()) +
                              list(self.bias_correction.parameters()),
                              lr=learning_rate)

    def get_nvil_cost(self, Y, n):
        batch_size = len(Y)
        q_hgys = []
        p_yhs = []
        C_outs = []
        for i in range(batch_size):
            for j in range(n):
                theta_samp = self.recognition_model.getSample(Y[i])
                # First, compute L and l (as defined in Algorithm 1 in Gregor & ..., 2014)
                # Evaluate the recognition model density Q_\phi(h_i | y_i)
                # print('hsamp fed into recognition model eval log density',hsamp)
                # print('Y fed into recognition model eval log density', Y)
                q_hgy = self.recognition_model.evalLogDensity(theta_samp, Y[i])
                # Evaluate the generative model density P_\theta(y_i , h_i)
                p_yh = self.generative_model.evaluateLogDensity(theta_samp, Y[i])
                # print('p_yh', p_yh)
                # print('Y', Y)
                # exact_posterior = self.generative_model.evaluateExactLogPosterior(Y)
                # print('exact_posterior:', exact_posterior)
                hidden0 = self.bias_correction.initHidden()
                input_degrees = Y[i]['degrees'].unsqueeze(0)
                input_degrees = input_degrees.permute(1, 0, 2)
                # print('input_degrees in bias correction',input_degrees)
                C_out = torch.squeeze(self.bias_correction.__call__(Variable(input_degrees), hidden0))
                q_hgys.append(q_hgy)
                p_yhs.append(p_yh)
                C_outs.append(C_out)
        p_yh = torch.stack(p_yhs)
        q_hgy = torch.stack(q_hgys)
        C_out = torch.stack(C_outs)
        # C_out = 0
        # print('C_out',C_out)
        L = p_yh.mean() - q_hgy.mean()
        # print('L',L)
        # print('q_hgy', q_hgy)
        # print('p_yh', p_yh)
        l = p_yh - q_hgy - C_out
        # print('p_yh',p_yh)
        # print('q_hgy',q_hgy)
        # print('C_out',C_out)
        # print('L', L)
        # print('l',l)
        return [L, l, p_yh, q_hgy, C_out]

    def update_cv(self, l):
        # Now compute derived quantities for the update
        # print(l)
        cb = l.mean().data
        vb = l.var().data
        # print('self.c', self.c)
        # print('self.v', self.v)
        self.c = self.alpha * self.c + (1-self.alpha) * cb
        self.v = self.alpha * self.v + (1-self.alpha) * vb
        # print('self.c after', self.c)
        # print('self.v after', self.v)

    def update_params(self, n, l, p_yh, q_hgy, C_out):
        self.opt.zero_grad()
        # print('l',l)
        # print('q_hgy:',q_hgy)
        loss_q = q_hgy[0]
        # print('loss_q:',loss_q)
        loss_C = C_out[0]
        for i in range(n):
            if torch.sqrt(self.v).numpy()[0] > 1.0:
                lii = (l[i].data - self.c) / torch.sqrt(self.v)
                # print('l[i].data000', l[i].data)
                # print('self.c000', self.c)
                # print('torch.sqrt(self.v)000',torch.sqrt(self.v))
                # print('lii000', lii)
            else:
                lii = l[i].data - self.c
                # print('l[i].data111',l[i].data)
                # print('self.c111',self.c)
                # print('lii111:', lii)
            lii = Variable(torch.FloatTensor(lii), requires_grad=False)

            if i == 0:
                loss_q = loss_q * lii * -1
                # print('loss_q1:',loss_q)
                # print('lii2',lii)
                loss_C = loss_C * lii * -1
            else:
                loss_q += q_hgy[i] * lii * -1
                # print('loss_q2:',loss_q)
                loss_C += C_out[i] * lii * -1
        loss_q = loss_q / n
        loss_C = loss_C / n
        # print('loss_q3:',loss_q)
        loss_q.backward(retain_graph=True)
        loss_C.backward(retain_graph=True)
        self.opt.step()
        # self.generative_model.update_pi()

    def save_model(self):
        model_path = settings.save_model_path
        torch.save(self.recognition_model, model_path + 'rec.model')
        torch.save(self.generative_model, model_path + 'gen.model')
        torch.save(self.bias_correction, model_path + 'bias.model')
        params = {'c': self.c, 'v': self.v, 'alpha': self.alpha,
                  'lr': self.learning_rate, 'num_classes': self.number_of_classes}
        pickle.dump(params, open(model_path + 'params.pkl', 'wb'))

    @staticmethod
    def load_model(model_path):
        rec_model = torch.load(model_path + 'rec.model')
        gen_model = torch.load(model_path + 'gen.model')
        bias_model = torch.load(model_path + 'bias.model')
        param = pickle.load(open(model_path + 'params.pkl', 'rb'))
        return NVIL(opt_params={'c0': param['c'], 'v0': param['v'], 'alpha': param['alpha']},
                    gen_params=None,
                    GEN_MODEL=gen_model,
                    REC_MODEL=rec_model,
                    number_of_classes=param['num_classes'],
                    learning_rate=param['lr'],
                    use_load_model=True,
                    bias_model=bias_model)


    def fit(self, dataset, batch_size=1, n=10, max_epochs=100, save=False):
        avg_costs = []
        epoch = 0
        while epoch < max_epochs:
            data_loader = NetworkIterator(dataset, batch_size=batch_size)
            sys.stdout.write("\r%0.2f%%\n" % (epoch * 100./ max_epochs))
            sys.stdout.flush()
            batch_counter = 0
            for data_x, data_y in data_loader:
                #print('data_x', data_x)
                #print('data_y', data_y)
                #hsamp_np = self.recognition_model.getSample(data_y)
                # print('hsamp_np',hsamp_np)
                L, l, p_yh, q_hgy, C_out = self.get_nvil_cost(data_y, n)
                self.update_cv(l)
                #print('data_y fed into update_params',data_y)
                self.update_params(n, l, p_yh, q_hgy, C_out)# data_y.shape[0] for batch_size
                if np.mod(batch_counter, 10) == 0:
                    cx = self.c
                    vx = self.v
                    print('(c, v, L): (%f, %f, %f)' % (np.asarray(cx), np.asarray(vx), L))
                avg_costs.append(L.data.numpy())
                batch_counter += 1
            epoch += 1
        if save:
            self.save_model()
        return avg_costs

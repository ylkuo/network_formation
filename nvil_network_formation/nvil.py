# Class for Neural Variational Inference and Learning (NVIL)

import numpy as np
import pickle, settings

if not settings.show_fig:
    import matplotlib
    matplotlib.use('Agg')

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from dataset import NetworkIterator
from matplotlib import pyplot as plt

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class Variable(torch.autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)


class Estimator():
    def __init__(self, rec_model, gen_model, n_samples=5, n_posterior_samples=10,
                 estimator_type='MAP', bin_size=5, which_posterior='variational',error_type='MSE'):
        self.rec_model = rec_model
        self.gen_model = gen_model
        self.n_samples = n_samples # number of samples generated for each true theta to test the estimator for
        # the given theta
        self.n_posterior_samples = n_posterior_samples # number of samples from the posterior toi construct an estimator
        self.estimator_type=estimator_type # estimator_type can be 'posterior_mean', 'MAP', 'median'
        self.bin_size = bin_size
        self.which_posterior = which_posterior # which_posterior can be 'exact', 'variational'
        self.error_type = error_type # error type can be MSE or MAE

    def get_estimates(self, theta, bin_size=5, do_hist=False):
        theta_estimates = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            y_val = dict().fromkeys(('network', 'degrees'))
            y_val['network'], y_val['degrees'] = self.gen_model.get_y(theta)
            sampled_thetas = np.zeros(self.n_posterior_samples)
            if self.which_posterior == 'variational':
                for j in range(self.n_posterior_samples):
                    sampled_theta = self.rec_model.getSample(y_val)
                    sampled_thetas[j] = sampled_theta.data[0]
            elif self.which_posterior == 'exact':
                sampled_thetas = self.gen_model.get_exact_posterior_samples(y_val,self.n_posterior_samples)
            else:
                assert False, "which_posterior is invalid."
            # print('sampled_thetas:',sampled_thetas)
            if do_hist:
                plt.hist(sampled_thetas)
                plt.title('Posterior Samples for theta = ' + str(theta))
                plt.show()
            # print(self.estimator_type)
            if self.estimator_type == 'posterior_mean':
                theta_estimates[i] = sampled_thetas.mean()
            elif self.estimator_type == 'MAP':
                hist, bin_edges = np.histogram(sampled_thetas, bins=bin_size)
                j = np.argmax(hist)
                theta_estimates[i] = (bin_edges[j] + bin_edges[j+1]) / 2.0
            elif self.estimator_type == 'median':
                theta_estimates[i] = np.median(sampled_thetas)
        error = 0
        if self.error_type == 'MSE':
            error = np.sum((selected_theta - theta)**2 for selected_theta in theta_estimates)
        elif self.error_type == 'MAE':
            error = np.sum(abs(selected_theta - theta) for selected_theta in theta_estimates)
        error /= self.n_samples
        return error, theta_estimates

    def get_estimates_for_true_thetas(self, true_thetas=[2,4,6], do_plot=True,
                                      symmetric=False, do_hist=False, verbose=True):
        estimated_thetas = [] # a list of theta_estimated for each true_theta
        mean_estimated_thetas = []
        errors = []
        for true_theta in true_thetas:
            if verbose:
                print('getting estimates for true theta:', true_theta)
            theta_error, theta_estimates = self.get_estimates(true_theta, do_hist=do_hist)
            estimated_thetas += [theta_estimates]
            mean_estimated_thetas += [np.mean(theta_estimates)]
            errors += [np.max(np.abs(theta_estimates - mean_estimated_thetas))]
            # TODO: need correction errors should be computed with respect to mean_estimated_thetas not true theta
            if verbose:
                print('true theta:', true_theta,'theta estimates:', theta_estimates)
        if do_plot:
            if symmetric:
                fig = plt.figure()
                plt.errorbar(true_thetas, mean_estimated_thetas, yerr=errors)
                plt.title(self.estimator_type + 'performance')
                plt.xlabel('true theta')
                plt.ylabel(self.estimator_type)
                if settings.show_fig: plt.show()
                if settings.save_fig: fig.savefig(save_model_path + 'posterior.png')
            else: # not symmetric
                lower_errors = []
                upper_errors = []
                for i in range(len(true_thetas)):
                    theta_estimates = estimated_thetas[i]
                    lower_errors += [abs(np.min(theta_estimates) - mean_estimated_thetas[i])]
                    upper_errors += [abs(np.max(theta_estimates) - mean_estimated_thetas[i])]
                asymmetric_errors = [lower_errors, upper_errors]
                fig = plt.figure()
                plt.errorbar(true_thetas, mean_estimated_thetas, yerr=asymmetric_errors)
                plt.title(self.estimator_type + ' performance ')
                plt.xlabel('true theta')
                plt.ylabel(self.estimator_type)
                if settings.show_fig: plt.show()
                if settings.save_fig: fig.savefig(settings.save_model_path + 'posterior.png')


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
        return (Variable(torch.zeros(self.num_layers, 1, self.hidden_size).type(dtype)),
                Variable(torch.zeros(self.num_layers, 1, self.hidden_size).type(dtype)))

    def getSampleOutput(self, torchSample):
        hidden0 = self.initHidden()
        # print(torchSample)
        if isinstance(torchSample, list) or isinstance(torchSample, tuple):
            feature = torchSample[1]
        else:
            feature = torchSample
        output = self.__call__(Variable(torch.from_numpy(np.asarray(feature)).type(dtype)), hidden0)

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
            self.recognition_model = REC_MODEL(self.number_of_classes).type(dtype)
            # print(gen_params)
            self.generative_model = GEN_MODEL(gen_params)

            # NVIL Bias-correction network
            self.bias_correction = bias_correction_RNN(input_size = settings.number_of_features, hidden_size = settings.n_hidden,
                     num_layers = settings.NUM_LAYERS, output_size = 1).type(dtype)
 
            # Set NVIL params
            self.c = torch.from_numpy(np.asarray([opt_params['c0']])).type(dtype)
            self.v = torch.from_numpy(np.asarray([opt_params['v0']])).type(dtype)
            self.alpha = torch.from_numpy(np.asarray([opt_params['alpha']])).type(dtype)

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
            if torch.sqrt(self.v).cpu().numpy()[0] > 1.0:
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
            if USE_CUDA: lii = lii.cuda()
            lii = Variable(lii, requires_grad=False)

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
                avg_costs.append(L.data.cpu().numpy())
                batch_counter += 1
            epoch += 1
        if save:
            self.save_model()
        return avg_costs

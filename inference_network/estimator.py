import numpy as np
import scipy.sparse as sp
import settings
import torch

if not settings.show_fig:
    import matplotlib
    matplotlib.use('Agg')

from dataset import normalize_adj
from generative_model import GenerativeModel
from matplotlib import pyplot as plt
from operator import itemgetter

class Estimator(object):
    def __init__(self, inference_network, n_samples=5, n_posterior_samples=10,
                 estimator_type='MAP', bin_size=5, which_posterior='variational',
                 error_type='MSE', do_sample=False):
        self.inference_network = inference_network
        self.n_samples = n_samples # number of samples generated for each true theta
        self.n_posterior_samples = n_posterior_samples # number of samples from the posterior toi construct an estimator
        self.estimator_type=estimator_type # estimator_type can be 'posterior_mean', 'MAP', 'median'
        self.bin_size = bin_size
        self.which_posterior = which_posterior # which_posterior can be 'exact', 'variational'
        self.error_type = error_type # error type can be MSE or MAE
        self.do_sample = do_sample # draw posterior samples or use the predicted distribution directly

    def get_estimates_for_theta(self, theta):
        model = GenerativeModel(settings.gen_model_params)
        theta_estimates = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            network = dict().fromkeys(('theta', 'in_sequence'))
            _, in_sequences, features = model.get_y(theta)
            seq_lengths = torch.LongTensor([len(in_sequences)])
            if settings.gen_model_params['input_type'] == 'adjacencies':
                norm_adj = []
                for adj in in_sequences:
                    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
                    norm_adj.append(np.array(adj.todense()))
                network['in_sequence'] = torch.tensor(norm_adj).type(settings.dtype)
            else:
                network['in_sequence'] = torch.tensor(in_sequences).type(settings.dtype)
            in_sequences = network['in_sequence'].unsqueeze(0)
            features = torch.tensor(features).type(settings.dtype)
            features = features.unsqueeze(0)
            if self.do_sample:
                # TODO: Add support to draw posterior samples
                pass
            else:
                proposal = self.inference_network.forward(features, in_sequences, seq_lengths)
                if self.estimator_type == 'posterior_mean':
                    theta_estimates[i] = proposal.mean_non_truncated.data[0]
                    print(theta, proposal.mean_non_truncated.data[0], proposal.stddev_non_truncated.data[0])
                else:
                    assert False, 'Do not support other estimate types if do_sample is False.'
        error = 0
        if self.error_type == 'MSE':
            error = np.sum((selected_theta - theta)**2 for selected_theta in theta_estimates)
        elif self.error_type == 'MAE':
            error = np.sum(abs(selected_theta - theta) for selected_theta in theta_estimates)
        error /= self.n_samples
        return error, theta_estimates

    def get_estimates_for_theta_on_data(self, theta, data):
        assert theta == data[1], "the data is not associated with the supplied theta"
        # model = GenerativeModel(settings.gen_model_params)
        theta_estimates = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            network = dict().fromkeys(('theta', 'in_sequence'))
            in_sequences, features = data[0],data[2]

            seq_lengths = torch.LongTensor([len(in_sequences)])
            if settings.gen_model_params['input_type'] == 'adjacencies':
                norm_adj = []
                for adj in in_sequences:
                    adj = normalize_adj(adj + sp.eye(adj.shape[0]))
                    norm_adj.append(np.array(adj.todense()))
                network['in_sequence'] = torch.tensor(norm_adj).type(settings.dtype)
            else:
                network['in_sequence'] = torch.tensor(in_sequences).type(settings.dtype)
            in_sequences = network['in_sequence'].unsqueeze(0)
            features = torch.tensor(features).type(settings.dtype)
            features = features.unsqueeze(0)
            if self.do_sample:
                # TODO: Add support to draw posterior samples
                pass
            else:
                proposal = self.inference_network.forward(features, in_sequences, seq_lengths)
                if self.estimator_type == 'posterior_mean':
                    theta_estimates[i] = proposal.mean_non_truncated.data[0]
                    print(theta, proposal.mean_non_truncated.data[0], proposal.stddev_non_truncated.data[0])
                else:
                    assert False, 'Do not support other estimate types if do_sample is False.'
        error = 0
        if self.error_type == 'MSE':
            error = np.sum((selected_theta - theta)**2 for selected_theta in theta_estimates)
        elif self.error_type == 'MAE':
            error = np.sum(abs(selected_theta - theta) for selected_theta in theta_estimates)
        error /= self.n_samples
        return error, theta_estimates

    def get_estimates(self, n_thetas, do_plot=True, symmetric=False, verbose=True):
        true_thetas = np.linspace(settings.gen_model_params['theta_range'][0],
                                  settings.gen_model_params['theta_range'][1],
                                  n_thetas)
        print(true_thetas)
        true_thetas.sort()
        print(true_thetas)
        estimated_thetas = []  # a list of theta_estimated for each true_theta
        mean_estimated_thetas = []
        errors = []
        for true_theta in true_thetas:
            if verbose:
                print('Getting estimates for true theta:', true_theta)
            theta_error, theta_estimates = self.get_estimates_for_theta(true_theta)
            estimated_thetas += [theta_estimates]
            mean_estimated_thetas += [np.mean(theta_estimates)]
            errors += [np.max(np.abs(theta_estimates - mean_estimated_thetas[-1]))]
            if verbose:
                print('true theta:', true_theta, 'theta estimates:', theta_estimates)
        if do_plot:
            if symmetric:
                fig = plt.figure()
                plt.errorbar(true_thetas, mean_estimated_thetas, yerr=errors)
                plt.title(self.estimator_type + 'performance')
                plt.xlabel('true theta')
                plt.ylabel(self.estimator_type)
                if settings.show_fig: plt.show()
                if settings.save_fig: fig.savefig(settings.model_prefix + 'posterior.png')
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
                if settings.save_fig: fig.savefig(settings.model_prefix + 'posterior.png')

    def get_estimates_on_data(self, data, do_plot=True, symmetric=False, verbose=True):
        data = sorted(data, key=itemgetter(1))
        true_thetas = [data[i][1] for i in range(len(data))]
        # print('sorted data theta', true_thetas)
        estimated_thetas = []  # a list of theta_estimated for each true_theta
        mean_estimated_thetas = []
        errors = []
        # print(len(data))
        for i in range(len(data)):
            print(data[i][1])

        for i in range(len(data)):
            if verbose:
                print('Getting estimates for true theta:', true_thetas[i])
            theta_error, theta_estimates = self.get_estimates_for_theta_on_data(true_thetas[i],data[i])
            estimated_thetas += [theta_estimates]
            mean_estimated_thetas += [np.mean(theta_estimates)]
            errors += [np.max(np.abs(theta_estimates - mean_estimated_thetas[-1]))]
            if verbose:
                print('true theta:', true_thetas[i], 'theta estimates:', theta_estimates)
        if do_plot:
            if symmetric:
                fig = plt.figure()
                plt.errorbar(true_thetas, mean_estimated_thetas, yerr=errors)
                plt.title(self.estimator_type + 'performance')
                plt.xlabel('true theta')
                plt.ylabel(self.estimator_type)
                if settings.show_fig: plt.show()
                if settings.save_fig: fig.savefig(settings.model_prefix + 'posterior.png')
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
                if settings.save_fig: fig.savefig(settings.model_prefix + 'posterior.png')

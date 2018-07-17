import pandas as pd
import copy
import torch
import math
import numpy as np
import random as RD
import networkx as NX
import torch.nn as nn
import settings

from torch.nn import functional as F

if settings.use_exact_posterior:
    import pymc as mc

USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
class Variable(torch.autograd.Variable):
    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

SENTINEL = object()

class NetworkModel(nn.Module):

    def __init__(self, network_params):
        super(NetworkModel, self).__init__()
        self.params = copy.deepcopy(network_params)
        self.fixed_network_params = copy.deepcopy(network_params)
        self.pairwise_stable = False

    def init_network(self):
        r"""
        Network structure is initialized here. Initial network should be supplied in params to analyze the Crunchbase
        dataset, because we need the nodes to be already labeled.
        """
        assert 'network' in self.fixed_network_params, 'The Crunchbase initial network is not supplied in params'
        self.params['network'] = copy.deepcopy(self.fixed_network_params['network'])
        if 'size' in self.fixed_network_params:
            assert self.params['size'] == NX.number_of_nodes(self.params['network']), 'network size mismatch'
        else:
            self.params['size'] = NX.number_of_nodes(self.params['network'])

        if 'input_type' not in self.fixed_network_params:
            self.params['input_type'] = 'degree_sequence'
            self.params['feature_length'] = self.params['size']

        if 'feature_length' not in self.fixed_network_params:
            if self.params['input_type'] == 'transitivity' or 'avg_clustering':
                self.params['feature_length'] = 1
            elif self.params['input_type'] == 'clustering' or 'degree_sequence':
                self.params['feature_length'] = self.params['size']
            else:
                assert False, 'mishandled type for training data'

    def init_network_attributes(self):
        r"""
        observable and unobservable node and edge attributes are initialized here
        """
        assert 'positions' in self.fixed_network_params, 'The Crunchbase node positions are not supplied in params'

        potential_edge_keys = list(self.params['network'].edges()) + list(NX.non_edges(self.params['network']))

        self.potential_edge_attributes = dict.fromkeys(potential_edge_keys)

        for i in self.params['network'].nodes():
            self.params['network'].node[i]['position'] = self.params['positions'][i]  # observable node attributes,
            # nodes are homophilic in the position attributes
            self.params['network'].node[i]['attribute'] = np.random.binomial(1, 0.5)  # will be removed,
            # may be latent homophilic attributes?

        for potential_edge in potential_edge_keys:
            self.potential_edge_attributes[potential_edge] = np.random.normal(0, 1)  # latent
            # noise variables that derive the link formation decisions

    def generate_time_series(self, utility_params, suply_network_timeseries=False):  # conditioned on the fixed_params
        self.init_network()
        self.set_utility_params(utility_params)
        self.init_network_attributes()

        network_timeseries = []
        while not self.pairwise_stable:
            dummy_network = self.params['network'].copy()
            network_timeseries.append(dummy_network)
            self.step()

        if self.params['input_type'] == 'network_timeseries':
            df = network_timeseries
        elif self.params['input_type'] == 'degree_sequence':
            all_nodes_degrees = list(
                map(lambda node_pointer: list(map(lambda network: 1.0 * (network.degree(node_pointer)),
                                                  network_timeseries)), self.params['network'].nodes()))
            df = pd.DataFrame(np.transpose(all_nodes_degrees))
        elif self.params['input_type'] == 'clustering':
            all_nodes_clustering = list(
                map(lambda node_pointer: list(map(lambda network: 1.0 * (NX.clustering(network, node_pointer)),
                                                  network_timeseries)), self.params['network'].nodes()))
            df = pd.DataFrame(np.transpose(all_nodes_clustering))
        elif self.params['input_type'] == 'transitivity':  # also called global clustering coeficient
            transitivity_timeseies = list(map(lambda network: 1.0 * (NX.transitivity(network)), network_timeseries))
            df = pd.DataFrame(transitivity_timeseies)
        elif self.params['input_type'] == 'avg_clustering':
            all_nodes_clustering = list(
                map(lambda node_pointer: list(map(lambda network: 1.0 * (NX.clustering(network, node_pointer)),
                                                  network_timeseries)), self.params['network'].nodes()))
            avg_clustering_timeseries = np.sum(all_nodes_clustering, 0) / self.params['size']
            df = pd.DataFrame(avg_clustering_timeseries)

        self.pairwise_stable = False

        if suply_network_timeseries:
            return df, network_timeseries
        else:
            return df

    def step(self):
        candidate_edges = list(NX.non_edges(self.params['network']))
        RD.shuffle(candidate_edges)
        self.pairwise_stable = True
        for candidate_edge in candidate_edges:
            # print(self.successful_edge_formation(candidate_edge).data)
            if self.successful_edge_formation(candidate_edge).data[0]:
                self.params['network'].add_edge(*candidate_edge)
                self.pairwise_stable = False
                break

    def set_utility_params(self):
        r"""
        Sets the parameters of the utility model.
        Each utility model has a different implemnetation for this function.
        """
        pass

    def successful_edge_formation(self, candidate_edge):
        r"""
        Computes the joint surplus of two nodes from forming a link with each other (the candidate_edge).
        Each utility model has a different implementation for this.
        """
        pass


class UtilityModel(NetworkModel):

    def __init__(self, network_params):
        super(UtilityModel, self).__init__(network_params)
        self.sparsity = nn.Linear(1, 1)
        self.sparsity.weight.data.fill_(0.1*np.sqrt(8 / self.params['size']))
        self.sparsity.bias.data.fill_(0.0)
        # print('self.sparsity.weight.data',self.sparsity.weight.data)
        # print('self.sparsity.weight.data', self.sparsity.bias.data)
        self.relative_weight = nn.Linear(1, 1)
        self.relative_weight.weight.data.fill_(0.10)
        self.relative_weight.bias.data.fill_(0.0)
        # print('self.relative_weight.weight.data', self.sparsity.weight.data)
        # print('self.relative_weight.bias.data', self.sparsity.bias.data)
        if USE_CUDA:
            self.sparsity = self.sparsity.cuda()
            self.relative_weight = self.relative_weight.cuda()

    def set_utility_params(self, utility_params):
        r"""
        Any thing that is not in the fixed_params will be randomized.
        These are the parameters of the utility model that are to be inferred from the observed networks
        """
        # print('utility_params',utility_params)
        # print(self.params)
        self.params.update(utility_params)
        # print('self.params',self.params)
        # x = self.params
        # self.params = x.update(utility_params)

        assert 'theta_2' in utility_params, "theta_2 is not set!"
            # self.params['theta_2'] = RD.choice(
            #     [self.params['lower_limit'], self.params['upper_limit']])  # np.random.normal(0, 1)#

        # if 'sparsity' not in utility_params:
        #     self.params['sparsity'] = 500 * np.sqrt(8 / self.params['size']) * 0.007

        # there should be better ways to set the parameters theta_0 and theta_3 (determining the sparsity) based on the
        # final edge density or something like that in the observed data.

    def successful_edge_formation(self, candidate_edge):
        #print('candidate_edge', candidate_edge)
        distance_risk_attitudes = np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][0] - \
                                                 self.params['network'].node[candidate_edge[1]]['position'][0])

        distance_investments = np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][1:] - \
                                              self.params['network'].node[candidate_edge[1]]['position'][1:])

        d_investments = torch.tensor([float(distance_investments)])
        if USE_CUDA: d_investments = d_investments.cuda()
        distance = F.relu(self.relative_weight(d_investments)) + float(distance_risk_attitudes)

        list_common_neighbors = list(
            NX.common_neighbors(self.params['network'], candidate_edge[0], candidate_edge[1]))
        #print('potential_edge_attributes', self.potential_edge_attributes[candidate_edge])
        #print('sparsity', self.sparsity.detach().numpy())
        pre_edge_value = self.params['theta_2']*(1.0 * (len(list_common_neighbors) > 0)) + \
            self.potential_edge_attributes[candidate_edge] - distance
        edge_value = self.sparsity(pre_edge_value).cpu().detach().numpy()
        # print(edge_value)
        edge_value = np.reshape(edge_value, 1)[-1]
        # print(edge_value.data)
        return edge_value > 0.0


class NetworkFormationGenerativeModel(UtilityModel):

    def __init__(self, GenerativeParams):

        super(NetworkFormationGenerativeModel, self).__init__(GenerativeParams)

        self.params['input_type'] = 'degree_sequence'

        # Mixture distribution
        # print('GenerativeParams', GenerativeParams)
        if 'prior' in GenerativeParams:
            self.prior_un = np.asarray(GenerativeParams['prior'])
        else:
            # self.prior_un = nn.Parameter(torch.FloatTensor(np.asarray(100*np.ones(settings.number_of_classes))), requires_grad=True)
            self.prior_un = np.linspace(settings.class_values[0]-1,settings.class_values[-1]+1,100)
            # what is the 100*??? should n't it be np.ones(xDim)
        self.prior = self.prior_un#/(settings.support)
        # print('self.pi in init NetFormationGenModel',self.pi)

    def get_y(self, theta):
        utility_params = dict().fromkeys(['theta_2'])
        utility_params['theta_2'] = theta
        degrees_df, networks = self.generate_time_series(utility_params, suply_network_timeseries=True)
        dummy1 = copy.copy(networks)
        #y_vals[ii]['network'] = copy.deepcopy(dummy1)
        dummy2 = copy.copy(torch.from_numpy(degrees_df.values[:, 0:self.params['feature_length']]).type(dtype))
        #y_vals[ii]['degrees'] = copy.copy(dummy2)
        return copy.deepcopy(dummy1), copy.copy(dummy2)

    def sampleXY(self, _N):
        _prior = self.prior
        # b_vals = np.random.multinomial(1, _prior, size=_N) # a binary vector of all zero entry except one (the chosen class)
        # x_vals = b_vals.nonzero()[1] # the index of the chosen class
        #
        # b_vals = np.asarray(b_vals)
        #
        y_vals = list(range(_N))

        theta_vals = list(range(_N))

        # print('x_vals before:',x_vals)

        for ii in range(_N):
            y_vals[ii] = dict().fromkeys(('network', 'degrees'))
            theta_vals[ii] = np.random.choice(_prior)#np.random.normal(mean_of_gaussain, 1)
            # print('theta_vals[ii]',theta_vals[ii])
            if theta_vals[ii] < 0:
                theta_vals[ii] = 0
            y_vals[ii]['network'], y_vals[ii]['degrees'] = self.get_y(theta_vals[ii])
        # print('b_vals:', b_vals)
        # print('x_vals after:', x_vals)
        # print('y_vals:', y_vals)
        return [theta_vals, y_vals]

    def normal_cdf(self, value):
        z = torch.div(value, math.sqrt(2))
        return 0.5 * (1 + torch.erf(z.type(dtype)))

    def non_edge_probability(self, non_edge, lastnetwork, theta_2):
        utility_params = dict.fromkeys(['theta_0','theta_1','theta_2','theta_3','sparsity'])
        utility_params['theta_0'] = 0
        utility_params['theta_2'] = theta_2
        utility_params['theta_3'] = 5
        utility_params['sparsity'] = 500 * np.sqrt(8 / self.params['size']) * 0.007
        self.set_utility_params(utility_params)

        distance_risk_attitudes = np.linalg.norm(lastnetwork.node[non_edge[0]]['position'][0] - \
                                                 lastnetwork.node[non_edge[1]]['position'][0])

        distance_investments = np.linalg.norm(lastnetwork.node[non_edge[0]]['position'][1:] - \
                                              lastnetwork.node[non_edge[1]]['position'][1:])

        d_investments = torch.tensor([float(distance_investments)])
        if USE_CUDA: d_investments = d_investments.cuda()
        distance = F.relu(self.relative_weight(d_investments)) + float(distance_risk_attitudes)
        list_common_neighbors = list(NX.common_neighbors(lastnetwork, non_edge[0], non_edge[1]))

        epsilon_upperbound = (self.sparsity.bias / self.sparsity.weight) \
                              + distance \
                              - theta_2 * (1.0 * (len(list_common_neighbors) > 0))

        probability_non_edge = self.normal_cdf(epsilon_upperbound)

        # print('probability_non_edge',probability_non_edge)

        return probability_non_edge

    def edge_probability(self, edge, network_time_series, lastnetwork, theta_2):
        # print('we are here')
        utility_params = dict.fromkeys(['theta_0', 'theta_1', 'theta_2', 'theta_3', 'sparsity'])
        utility_params['theta_0'] = 0
        utility_params['theta_2'] = theta_2
        utility_params['theta_3'] = 5
        utility_params['sparsity'] = 500 * np.sqrt(8 / self.params['size']) * 0.007
        self.set_utility_params(utility_params)

        distance_risk_attitudes = np.linalg.norm(lastnetwork.node[edge[0]]['position'][0] - \
                                                 lastnetwork.node[edge[1]]['position'][0])

        distance_investments = np.linalg.norm(lastnetwork.node[edge[0]]['position'][1:] - \
                                              lastnetwork.node[edge[1]]['position'][1:])

        d_investments = torch.tensor([float(distance_investments)])
        if USE_CUDA: d_investments = d_investments.cuda()
        distance = F.relu(self.relative_weight(d_investments)) + float(distance_risk_attitudes)

        list_common_neighbors = list(NX.common_neighbors(lastnetwork, edge[0], edge[1]))

        # find the time of formation:
        formation_time = 0
        for i in range(len(network_time_series)):
            # print('i',i)
            if edge in NX.edges(network_time_series[i]):  # edge is formed at time i
                # print('true',i)
                formation_time = i
                break

        # print(formation_time)
        # print(edge)
        # print(lastnetwork.edges())
        #
        # print(network_time_series[0].edges())

        if len(list_common_neighbors) > 0:
            # find the common_neighbor_time:
            common_neighbor_time = 0
            for i in range(len(network_time_series)):
                list_common_neighbors = list(NX.common_neighbors(network_time_series[i], edge[0], edge[1]))
                if len(list_common_neighbors) > 0:  # common_neighbor exists at time i
                    common_neighbor_time = i
                    break
        else:  # common_neighbors never obtained
            common_neighbor_time = len(network_time_series)+1

        assert common_neighbor_time != formation_time, "three edges of a triangle are formed at the same time!!!"

        if formation_time == 0:  # initial edges
            probability_of_the_edge = torch.from_numpy(np.asarray([1.0])).type(dtype)
            # the initial edges are there with probability one (computations are conditioned
            # on the initial condition)
            # print('probability_of_the_edge case 1', probability_of_the_edge)

        elif len(list_common_neighbors) == 0 or common_neighbor_time > formation_time:
            # no common neighbor at the time of link formation decision
            product_term = 1
            for i in range(formation_time):
                if i < formation_time -1:
                    number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                    product_term *= (1 - 1 / number_of_non_edges)
                elif i == formation_time -1:
                    number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                    product_term *= (1 / number_of_non_edges)

            epsilon_upperbound = -self.sparsity.bias / self.sparsity.weight + distance
            probability_of_the_edge = product_term*self.normal_cdf(epsilon_upperbound)

            # print('product_term',product_term)
            # print('probability_of_the_edge case 2',probability_of_the_edge)

        elif common_neighbor_time < formation_time:

            first_product_term = 1
            for i in range(formation_time):
                if i < formation_time -1:
                    number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                    first_product_term *= (1 - 1 / number_of_non_edges)
                elif i == formation_time -1:
                    number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                    first_product_term *= (1 / number_of_non_edges)

            second_product_term = 1
            for i in range(common_neighbor_time, formation_time):
                if i < formation_time -1:
                    number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                    second_product_term *= (1 - 1 / number_of_non_edges)
                elif i == formation_time -1:
                    number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                    second_product_term *= (1 / number_of_non_edges)


            epsilon_lowerbound = -self.sparsity.bias / self.sparsity.weight \
                                 + distance - theta_2
            # it has too many [[[[[]]]]]

            epsilon_upperbound = -self.sparsity.bias / self.sparsity.weight + distance

            probability_of_the_edge = (second_product_term * (self.normal_cdf(epsilon_upperbound) -
                                                              self.normal_cdf(epsilon_lowerbound))) + \
                                      (first_product_term * (1 - self.normal_cdf(epsilon_upperbound)))

            # print('first_product_term', first_product_term)
            # print('second_product_term', second_product_term)

            # print('probability_of_the_edge case 3',probability_of_the_edge)

        return probability_of_the_edge

    def evaluateLogDensity(self, theta_val, Y):
        # print('theta_val',theta_val)

        # print('Y',Y)
        # print('theta_val',theta_val)
        # X = torch.t(h.nonzero())[1]
        # print('X',X)
        #log_density = []
        #for count in range(1): #range(Y.shape[0]):
        X = theta_val
        # print(X)
        network_time_series = Y['network'] #[count]['network']
        last_network = network_time_series[-1]
        unformed_edges = NX.non_edges(last_network)
        formed_edges = NX.edges(last_network)
        LogDensityVeci = 0
        total_edges = 0

        for non_edge in unformed_edges:
            LogDensityVeci += torch.log(self.non_edge_probability(non_edge, last_network, X)) # X[count]
            # print('non_edge',LogDensityVeci)
            total_edges += 1

        for edge in formed_edges:
            # print(self.edge_probability(edge, network_time_series, last_network, X))
            LogDensityVeci += torch.log(self.edge_probability(edge, network_time_series, last_network, X)) # X[count]
            # print('edge',LogDensityVeci)
            total_edges += 1

        # print('pi',self.pi)
        # print('X',X)
        # print('pi', self.pi)
        # print('self.pi[X]', self.pi[X])
        log_density = (LogDensityVeci +
                       torch.log(Variable(torch.from_numpy(np.asarray([1/settings.support])).type(dtype)
                                          , requires_grad=True))) #  #/total_edges # /total_edges X[count]

        # print('log_density',log_density)
        if math.isnan(log_density):
            print('theta_val:', theta_val)
        assert not math.isnan(log_density), "log_density is nan!!!"
        return log_density #torch.squeeze(torch.stack(log_density))

    def get_log_marginal_probability(self, Y, n_samples=150):
        '''Use numerical integration to get the marginal probability of the observed data.'''
        # written for uniform prior can be extended to other priors.
        assert n_samples<=150, "large n_samples lead to numerical errors"
        uniform_prior = 1 / settings.support

        integration_points = np.linspace(settings.class_values[0], settings.class_values[-1], n_samples)

        log_prior = [np.log(uniform_prior)] * len(integration_points)

        # print('log_prior',log_prior)

        log_likelihood = np.asarray([self.evaluateLogDensity(torch.tensor([integration_points[jj]]).type(dtype), Y)
                                     for jj in range(n_samples)])

        # print('log_likelihood', log_likelihood)

        # print('log_likelihood',log_likelihood)

        marginal_probability_points = np.exp(log_prior + log_likelihood)

        numerical_integration_width = settings.support/n_samples

        # print('marginal_probability_points', marginal_probability_points)

        marginal_probability = np.sum(marginal_probability_points)*numerical_integration_width

        # print('marginal_probability', marginal_probability)
        # print(np.log(marginal_probability.data))

        return np.log(marginal_probability.data)


    def get_exact_posterior_samples(self, Y,n_samples=100):
        '''Use MCMC to get samples from the exact posterior. '''

        infer_theta = mc.Uniform('infer_theta', settings.class_values[0],
                                   settings.class_values[-1])  # this is the prior on the quality

        data = Y['network']

        @mc.stochastic(observed=True)
        def network_likelihood_model(value=data,infer_theta=infer_theta):
            # print('infer_theta:', infer_theta.tolist())
            X = Variable(torch.from_numpy(np.asarray([infer_theta.tolist()])).type(dtype),requires_grad=False)
            # print('data:', data)
            # print('value:',value)
            network_time_series = data
            last_network = network_time_series[-1]
            unformed_edges = NX.non_edges(last_network)
            formed_edges = NX.edges(last_network)
            LogDensityVeci = 0
            total_edges = 0
            for non_edge in unformed_edges:
                LogDensityVeci += torch.log(self.non_edge_probability(non_edge, last_network, X))  # X[count]
                total_edges += 1
            for edge in formed_edges:
                LogDensityVeci += torch.log(
                    self.edge_probability(edge, network_time_series, last_network, X))  # X[count]
                total_edges += 1
            # print('pi',self.pi)
            # print('X',X)
            # print('self.pi', self.pi)
            # print('self.pi[X]', self.pi[0])
            log_likelihood = LogDensityVeci + torch.log(Variable(torch.from_numpy(np.asarray([1/settings.support])).type(dtype)
                                                                 ,requires_grad=False))
            # log_densities = torch.squeeze(torch.log(torch.div(torch.stack(numerators), denominator)))
            # print('log_likelihood',log_likelihood)
            return log_likelihood.data.cpu().numpy()

        posterior_samples = np.zeros(n_samples)

        #  MCMC

        #  the MH alg is run for iter=10000 times
        #  the  first burn=2000 samples are dumped and from that point on every thin=100 sample one is taken
        #  thin is to avoid correlation among samples.

        thin = 100
        burn_in = 2000
        MH_iter = thin*(n_samples) + burn_in

        # for i in range(n_samples):
        #     print('i:',i)
        model = mc.MCMC([infer_theta, network_likelihood_model])
        model.sample(iter=MH_iter, burn=burn_in, thin=thin, progress_bar=False)
        posterior_samples = model.trace('infer_theta')[:]
        # print(print('posterior_samples inside MCMC shape', posterior_samples.shape))
        # print('posterior_samples inside MCMC', posterior_samples)

        # posterior_mean = np.mean(model.trace('infer_theta')[:])
        return posterior_samples  # torch.squeeze(torch.stack(log_density))

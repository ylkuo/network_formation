import pandas as pd
import copy
import torch
import math
import numpy as np
import random as RD
import networkx as NX
import torch.nn as nn
from torch.autograd import Variable
import settings

RD.seed()
np.random.seed()

SENTINEL = object()



class NetworkModel:

    def __init__(self, network_params):
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
            self.potential_edge_attributes[potential_edge] = np.random.normal(0, self.params['theta_3'])  # latent
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
            if self.successful_edge_formation(candidate_edge):
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
        # print(self.params)
        if 'theta_0' not in utility_params:
            self.params['theta_0'] = 0  # np.random.normal(0, 1)
        #if 'theta_1' not in utility_params:
        #    self.params['theta_1'] = 0  # np.random.normal(0, 1)

        assert 'theta_2' in utility_params, "theta_2 is not set!"
            # self.params['theta_2'] = RD.choice(
            #     [self.params['lower_limit'], self.params['upper_limit']])  # np.random.normal(0, 1)#

        if 'theta_3' not in utility_params:
            self.params['theta_3'] = 1  # np.random.normal(0, 1)
        if 'sparsity' not in utility_params:
            self.params['sparsity'] = 500 * np.sqrt(8 / self.params['size'])
        # there should be better ways to set the parameters theta_0 and theta_3 (determining the sparsity) based on the
        # final edge density or something like that in the observed data.

    def successful_edge_formation(self, candidate_edge):
        distance_risk_attitudes = np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][0] - \
                                                 self.params['network'].node[candidate_edge[1]]['position'][0])

        distance_investmnets = np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][1:] - \
                                              self.params['network'].node[candidate_edge[1]]['position'][1:])

        distance = 0.1 * distance_investmnets + distance_risk_attitudes

        list_common_neighbors = list(
            NX.common_neighbors(self.params['network'], candidate_edge[0], candidate_edge[1]))
        edge_value = self.params['theta_0'] + \
                     self.params['theta_2'] * (1.0 * (len(list_common_neighbors) > 0)) + \
                     self.potential_edge_attributes[candidate_edge] - \
                     (1 / self.params['sparsity']) * distance

        return edge_value > 0


class NetworkFormationGenerativeModel(UtilityModel):

    def __init__(self, GenerativeParams):

        super(NetworkFormationGenerativeModel, self).__init__(GenerativeParams)

        self.params['input_type'] = 'degree_sequence'

        # Mixture distribution
        # print('GenerativeParams', GenerativeParams)
        if 'pi' in GenerativeParams:

            self.pi_un = nn.Parameter(torch.FloatTensor(np.asarray(GenerativeParams['pi'])), requires_grad=True)
        else:
            self.pi_un = nn.Parameter(torch.FloatTensor(np.asarray(100*np.ones(settings.number_of_classes))), requires_grad=True)
            # what is the 100*??? should n't it be np.ones(xDim)
        self.pi = self.pi_un / self.pi_un.sum()
        # print('self.pi in init NetFormationGenModel',self.pi)

    def sampleXY(self, _N):
        _pi = np.asarray(torch.clamp(self.pi, 0.001, 0.999).data)

        b_vals = np.random.multinomial(1, _pi, size=_N) # a binary vector of all zero entry except one (the chosen class)
        x_vals = b_vals.nonzero()[1] # the index of the chosen class

        b_vals = np.asarray(b_vals)

        y_vals = list(range(_N))

        # print(y_vals)

        for ii in range(_N):
            y_vals[ii] = dict().fromkeys(('network', 'degrees'))
            utility_params = dict().fromkeys(['theta_2'])
            utility_params['theta_2'] = x_vals[ii]
            degrees_df, networks = self.generate_time_series(utility_params,suply_network_timeseries=True)
            dummy1 = copy.copy(networks)
            y_vals[ii]['network'] = copy.deepcopy(dummy1)
            # y_vals[ii]['network'] is used only to evaluate the log-densities it is not supplies as input to
            # the neural networks
            dummy2 = copy.copy(torch.FloatTensor(degrees_df.values[:, 0:self.params['feature_length']]))
            y_vals[ii]['degrees'] = copy.copy(dummy2)
            # print('y_vals[ii][degrees]',y_vals[ii]['degrees'])
            # print(y_vals)
        # print(b_vals, y_vals)
        return [b_vals, y_vals]

    def parameters(self):
        params_list = [self.pi_un]
        for params in params_list:
            yield params

    def update_pi(self):
        #print('self.pi in update_pi',self.pi)
        self.pi = self.pi_un / self.pi_un.sum()

    def normal_cdf(self, value):
        # print(value.type(torch.FloatTensor))
        # print('value', value.data)
        # print('value',value)
        z = torch.div(value,math.sqrt(2))#.type(torch.FloatTensor)
        # print('z', z)
        # z = value/math.sqrt(2)
        # print('z',z)
        return 0.5 * (1 + torch.erf(z.type(torch.FloatTensor)))
        #0.5 * (1 + torch.erf(torch.FloatTensor([value /math.sqrt(2)])))

    def non_edge_probability(self, non_edge, lastnetwork, theta_2):
        utility_params = dict.fromkeys(['theta_0','theta_1','theta_2','theta_3','sparsity'])
        utility_params['theta_0'] = 0
        utility_params['theta_2'] = theta_2
        utility_params['theta_3'] = 1
        utility_params['sparsity'] = 500 * np.sqrt(8 / 20)
        self.set_utility_params(utility_params)

        distance_risk_attitudes = np.linalg.norm(lastnetwork.node[non_edge[0]]['position'][0] - \
                                                 lastnetwork.node[non_edge[1]]['position'][0])

        distance_investments = np.linalg.norm(lastnetwork.node[non_edge[0]]['position'][1:] - \
                                              lastnetwork.node[non_edge[1]]['position'][1:])

        distance = 0.1 * distance_investments + distance_risk_attitudes
        list_common_neighbors = list(NX.common_neighbors(lastnetwork, non_edge[0], non_edge[1]))
        epsilon_upperbound = - self.params['theta_0'] - \
                             theta_2 * (1.0 * (len(list_common_neighbors) > 0)) + \
                             (1 / self.params['sparsity']) * distance

        # print(epsilon_upperbound)
        return self.normal_cdf(epsilon_upperbound)#.type(torch.FloatTensor)

    def edge_probability(self, edge, network_time_series, lastnetwork, theta_2):
        utility_params = dict.fromkeys(['theta_0', 'theta_1', 'theta_2', 'theta_3', 'sparsity'])
        utility_params['theta_0'] = 0
        utility_params['theta_2'] = theta_2
        utility_params['theta_3'] = 1
        utility_params['sparsity'] = 500 * np.sqrt(8 / 20)
        self.set_utility_params(utility_params)

        distance_risk_attitudes = np.linalg.norm(lastnetwork.node[edge[0]]['position'][0] - \
                                                 lastnetwork.node[edge[1]]['position'][0])

        distance_investments = np.linalg.norm(lastnetwork.node[edge[0]]['position'][1:] - \
                                              lastnetwork.node[edge[1]]['position'][1:])

        distance = 0.1 * distance_investments + distance_risk_attitudes

        list_common_neighbors = list(NX.common_neighbors(lastnetwork, edge[0], edge[1]))

        if edge in NX.edges(network_time_series[0]):
            probability_of_the_edge = 1 # the initial edges are there with probability one (computations are conditioned
            # on the initial condition)
        elif len(list_common_neighbors) == 0:
            epsilon_upperbound = - self.params['theta_0'] + (1 / self.params['sparsity']) * distance
            # print('epsilon_upperbound:', epsilon_upperbound,epsilon_upperbound.shape)
            epsilon_upperbound = Variable(torch.FloatTensor([epsilon_upperbound]))
            # print('epsilon_upperbound:', epsilon_upperbound,epsilon_upperbound.shape)
            probability_of_the_edge = self.normal_cdf(epsilon_upperbound) # needs to be fixed!!
        elif len(list_common_neighbors) > 0:
            product_term = 1
            for i in range(len(network_time_series)):
                # print('network_time_series')
                # print(len(list(NX.non_edges(network_time_series[i]))))
                number_of_non_edges = len(list(NX.non_edges(network_time_series[i])))
                if edge in NX.non_edges(network_time_series[i]):
                    product_term *= (1 - 1 / number_of_non_edges)
                else:
                    break

            # print('theta_2',theta_2)
            epsilon_lowerbound = - self.params['theta_0'] + (1 / self.params['sparsity']) * distance - theta_2 * (1.0)
            # it has too many [[[[[]]]]]
            # print(epsilon_lowerbound,epsilon_lowerbound.shape)
            epsilon_lowerbound = np.reshape(epsilon_lowerbound, 1)[-1]  # taking epsilon_lowerbound out of [[[[[]]]]]

            # print(epsilon_lowerbound,epsilon_lowerbound.shape)

            epsilon_upperbound = Variable(torch.FloatTensor([- self.params['theta_0'] +
                                                             (1 / self.params['sparsity']) * distance]))

            # print('epsilon_lowerbound', epsilon_lowerbound)  # it has too many [[[[[]]]]]
            # print('epsilon_upperbound', epsilon_upperbound)   # needed to be wrapped inside a variable
            probability_of_the_edge = (self.normal_cdf(epsilon_upperbound) - self.normal_cdf(epsilon_lowerbound)) + \
                                      product_term * (1 - self.normal_cdf(epsilon_upperbound))
        return probability_of_the_edge

    def evaluateLogDensity(self, h, Y):
        # print('Y',Y)
        # print('h',h)
        X = torch.t(h.nonzero())[1]
        # print('X',X)
        #log_density = []
        #for count in range(1): #range(Y.shape[0]):
        network_time_series = Y['network'] #[count]['network']
        last_network = network_time_series[-1]
        unformed_edges = NX.non_edges(last_network)
        formed_edges = NX.edges(last_network)
        LogDensityVeci = 0
        total_edges = 0

        for non_edge in unformed_edges:
            LogDensityVeci += torch.log(self.non_edge_probability(non_edge, last_network, X)) # X[count]
            total_edges += 1

        for edge in formed_edges:
            LogDensityVeci += torch.log(self.edge_probability(edge, network_time_series, last_network, X)) # X[count]
            total_edges += 1

        # print('pi',self.pi)
        # print('X',X)
        # print('pi', self.pi)
        # print('self.pi[X]', self.pi[X])
        log_density = (LogDensityVeci + torch.log(self.pi[X]))#/total_edges # /total_edges X[count]

        return log_density #torch.squeeze(torch.stack(log_density))

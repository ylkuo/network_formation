# the generative model of network formation

import pandas as pd
import copy
import torch
import numpy as np
import random as RD
import networkx as NX
import matplotlib.pyplot as plt
import pickle


RD.seed()
np.random.seed()

SENTINEL = object()


class NetworkModel:

    def __init__(self,params):
        self.params = copy.deepcopy(params)
        self.fixed_params = copy.deepcopy(params)
        self.pairwise_stable = False

    def init_network(self):
        r"""
        Network structure is initialized here. Initial network should be supplied in params to analyze the Crunchbase
        dataset, because we need the nodes to be already labeled.
        """
        assert 'network' in self.fixed_params, 'The Crunchbase initial network is not supplied in params'
        self.params['network'] = copy.deepcopy(self.fixed_params['network'])
        if 'size' in self.fixed_params:
            assert self.params['size'] == NX.number_of_nodes(self.params['network']), 'network size mismatch'
        else:
            self.params['size'] = NX.number_of_nodes(self.params['network'])

        if 'input_type' not in self.fixed_params:
            self.params['input_type'] = 'transitivity'
            self.params['feature_length'] = 1

        if 'feature_length' not in self.fixed_params:
            if self.params['input_type'] == 'transitivity' or 'avg_clustering':
                self.params['feature_length'] = 1
            elif self.params['input_type'] == 'clustering':
                self.params['feature_length'] = self.params['size']
            else:
                assert False, 'mishandled type for training data'

    def init_network_attributes(self):
        r"""
        observable and unobservable node and edge attributes are initialized here
        """
        assert 'positions' in self.fixed_params, 'The Crunchbase node positions are not supplied in params'

        potential_edge_keys = list(self.params['network'].edges()) + list(NX.non_edges(self.params['network']))

        self.potential_edge_attributes = dict.fromkeys(potential_edge_keys)

        for i in self.params['network'].nodes():
            self.params['network'].node[i]['position'] = self.params['positions'][i]  # observable node attributes,
        # nodes are homophilic in the position attributes
            self.params['network'].node[i]['attribute'] = np.random.binomial(1, 0.5) # will be removed,
            # may be latent homophilic attributes?

        for potential_edge in potential_edge_keys:
            self.potential_edge_attributes[potential_edge] = np.random.normal(0, self.params['theta_3'])  # latent
            # noise variables that derive the link formation decisions


    def generate_time_series(self):  # conditioned on the fixed_params
        self.init_network()
        self.set_random_params()
        self.init_network_attributes()

        network_timeseries = []

        while not self.pairwise_stable:
            dummy_network = self.params['network'].copy()
            network_timeseries.append(dummy_network)
            self.step()

        if self.params['input_type'] == 'clustering':
            all_nodes_clustering = list(
                map(lambda node_pointer: list(map(lambda network: 1.0 * (NX.clustering(network,node_pointer)),
                                                  network_timeseries)), self.params['network'].nodes()))
            df = pd.DataFrame(np.transpose(all_nodes_clustering))
        elif self.params['input_type'] == 'transitivity': # also called global clustering coeficient
            transitivity_timeseies = list(map(lambda network: 1.0 * (NX.transitivity(network)),network_timeseries))
            df = pd.DataFrame(transitivity_timeseies)
        elif self.params['input_type'] == 'avg_clustering':
            all_nodes_clustering = list(
                map(lambda node_pointer: list(map(lambda network: 1.0 * (NX.clustering(network, node_pointer)),
                                                  network_timeseries)), self.params['network'].nodes()))
            avg_clustering_timeseries = np.sum(all_nodes_clustering , 0) / self.params['size']
            df = pd.DataFrame(avg_clustering_timeseries)

        self.pairwise_stable = False

        return df

    def plot_time_series(self):
        df = self.generate_time_series()
        if self.params['input_type'] == 'transitivity':
            df.columns = ['transitivity (global clustering coeficient)']
            df['new links added'] = pd.Series(list(range(len(df))))
            df.plot(x='new links added', y='transitivity (global clustering coeficient)')
            plt.show()
        elif self.params['input_type'] == 'avg_clustering':
            df.columns = ['average clustering']
            df['new links added'] = pd.Series(list(range(len(df))))
            df.plot(x='new links added', y='average clustering')
            plt.show()

    def step(self):
        candidate_edges = list(NX.non_edges(self.params['network']))
        RD.shuffle(candidate_edges)
        self.pairwise_stable = True
        for candidate_edge in candidate_edges:
            if self.successful_edge_formation(candidate_edge):
                self.params['network'].add_edge(*candidate_edge)
                self.pairwise_stable = False
                break

    def gen_torch_sample(self):
        df = self.generate_time_series()
        #print(df)
        #print(df.values[:, 0:self.params['feature_length']])
        data = torch.FloatTensor(df.values[:, 0:self.params['feature_length']])
        # print(data)
        label = int(self.params['theta_2']>0)
        label_of_data = torch.LongTensor([label])
        return label_of_data, data

    def gen_torch_data_set(self, dataset_size=1000,filename = 'dataset.pkl', LOAD=False, SAVE = False ):
        if LOAD:
            simulation_results = pickle.load(open('./data/'+filename,'rb'))
        else:
            simulation_results = []
            for iter in range(dataset_size):
                label_of_data, data = self.gen_torch_sample()
                # print(data)
                simulation_results.append((label_of_data, data))
            if SAVE:
                pickle.dump(simulation_results, open('./data/'+filename, 'wb'))

        return simulation_results

    def set_random_params(self):
        r"""
        Sets the parameters of the utility model.
        Each utility model has a different implemnetation for this function.
        """
        pass

    def successful_edge_formation(self,candidate_edge):
        r"""
        Computes the joint surplus of two nodes from forming a link with each other (the candidate_edge).
        Each utility model has a different implementation for this.
        """
        pass


class UtilityModel(NetworkModel):

    def __init__(self,params):
        super(UtilityModel,self).__init__(params)

    def set_random_params(self):
        r"""
        These are the parameters of the utility model that are to be inferred from the observed networks
        """
        if 'theta_0' not in self.fixed_params:
            self.params['theta_0'] = 0  # np.random.normal(0, 1)
        if 'theta_1' not in self.fixed_params:
            self.params['theta_1'] = 0  # np.random.normal(0, 1)
        if 'theta_2' not in self.fixed_params:
            self.params['theta_2'] = RD.choice([self.params['lower_limit'], self.params['upper_limit']])  # np.random.normal(0, 1)#
        if 'theta_3' not in self.fixed_params:
            self.params['theta_3'] = 1  # np.random.normal(0, 1)
        if 'sparsity' not in self.fixed_params:
            self.params['sparsity'] = 500*np.sqrt(8 / self.params['size'])
        # there should be better ways to set the parameters theta_0 and theta_3 (determining the sparsity) based on the
        # final edge density or something like that in the observed data.


    def successful_edge_formation(self,candidate_edge):
        distance_risk_attitudes = np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][0] - \
                                  self.params['network'].node[candidate_edge[1]]['position'][0])

        distance_investmnets = np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][1:] - \
                                                 self.params['network'].node[candidate_edge[1]]['position'][1:])

        distance = 0.1*distance_investmnets + distance_risk_attitudes

        # print('distance_risk_attitudes', distance_risk_attitudes)
        # print('distance', distance)
        # print('distance_investmnets', distance_investmnets)

        list_common_neighbors = list(NX.common_neighbors(self.params['network'], candidate_edge[0], candidate_edge[1]))
        edge_value = self.params['theta_0'] + \
                     self.params['theta_2']*(1.0*(len(list_common_neighbors)>0)) + \
                     self.potential_edge_attributes[candidate_edge] - \
                     (1/self.params['sparsity'])*distance

        # self.params['theta_1'] * \
        # (self.params['network'].node[candidate_edge[0]]['attribute'] + \
        #  self.params['network'].node[candidate_edge[1]]['attribute']) + \

        # print('theta_0',self.params['theta_0'])
        #
        # print('theta_2',self.params['theta_2'])
        #
        # print('common_neighbors', (1.0 * (len(list_common_neighbors) > 0)))
        #
        # print('edge',self.potential_edge_attributes[candidate_edge])
        #
        # print('distance',(1/self.params['sparsity'])*distance)
        #
        # print('edge_value',edge_value)
        return edge_value > 0
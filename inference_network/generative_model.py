import copy
import networkx as nx
import numpy as np
import os
import pandas as pd
import pylab as pl
import random as rd
import settings

class GenerativeModel(object):
    def __init__(self, params):
        self.params = copy.deepcopy(params)
        self.pairwise_stable = False

    def init_network(self):
        # initialize network and positions
        self.params['network'] = nx.empty_graph(self.params['size'])
        self.params['positions'] = nx.random_layout(self.params['network'])
        # set input data type to be degree sequence
        if 'input_type' not in self.params:
            self.params['input_type'] = 'degree_sequence'
            self.params['feature_length'] = self.params['size']
        # TODO: init for other data type

    def init_network_attributes(self):
        potential_edge_keys = list(self.params['network'].edges()) + \
            list(nx.non_edges(self.params['network']))
        self.potential_edge_attributes = dict.fromkeys(potential_edge_keys)
        # set node attributes
        for i in self.params['network'].nodes():
            # nodes are homophilic in the position attributes
            self.params['network'].node[i]['position'] = self.params['positions'][i]
        # set edge attributes
        for potential_edge in potential_edge_keys:
            # noise variables that derive the link formation decisions
            # try (0,0.01)
            self.potential_edge_attributes[potential_edge] = np.random.normal(0, 1)  # latent

    def set_utility_params(self, utility_params):
        r"""
        These are the parameters of the utility model that are to be inferred from the observed networks
        """
        self.params.update(utility_params)

    def step(self):
        candidate_edges = list(nx.non_edges(self.params['network']))
        rd.shuffle(candidate_edges)
        self.pairwise_stable = True
        for candidate_edge in candidate_edges:
            if self.successful_edge_formation(candidate_edge):
                self.params['network'].add_edge(*candidate_edge)
                self.pairwise_stable = False
                break

    def draw(self, t):
        pl.cla()
        nx.draw(self.params['network'],
                pos=self.params['positions'],
                node_color=[0 for i in self.params['network'].nodes()],
                with_labels=False,
                edge_color='c',
                cmap=pl.cm.YlOrRd,
                vmin=0,
                vmax=1)
        pl.axis('image')
        pl.title('t = ' + str(t))
        img_path = settings.img_prefix + '/theta_' + str(self.params['theta'])
        pl.savefig(img_path + '/' + str(t) + '.png')

    def generate_time_series(self, utility_params, suply_network_timeseries=False):
        self.init_network()
        self.set_utility_params(utility_params)
        self.init_network_attributes()
        if settings.save_network_img:
            img_path = settings.img_prefix + '/theta_' + str(self.params['theta'])
            if not os.path.isdir(img_path):
                os.mkdir(img_path)
        # generate time series
        network_timeseries = []
        t = 0
        while not self.pairwise_stable:
            dummy_network = self.params['network'].copy()
            network_timeseries.append(dummy_network)
            self.step()
            t += 1
            if settings.save_network_img:
                self.draw(t)
        # turning time series to degree sequence
        if self.params['input_type'] == 'degree_sequence':
            all_nodes_degrees = list(
                map(lambda node_pointer: list(map(lambda network: 1.0 * (network.degree(node_pointer)),
                                                  network_timeseries)), self.params['network'].nodes()))
            df = pd.DataFrame(np.transpose(all_nodes_degrees))
        self.pairwise_stable = False

        if suply_network_timeseries:
            return df, network_timeseries
        else:
            return df

    def successful_edge_formation(self, candidate_edge):
        distance_risk_attitudes = \
            np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][0] - \
                           self.params['network'].node[candidate_edge[1]]['position'][0])

        distance_investments = \
            np.linalg.norm(self.params['network'].node[candidate_edge[0]]['position'][1:] - \
                           self.params['network'].node[candidate_edge[1]]['position'][1:])
        distance = 0.1 * distance_investments + distance_risk_attitudes
        list_common_neighbors = list(
            nx.common_neighbors(self.params['network'], candidate_edge[0], candidate_edge[1]))
        edge_value = (1 / self.params['sparsity']) * \
                     (self.params['theta'] * (1.0 * (len(list_common_neighbors) > 0)) - distance) + \
                      self.potential_edge_attributes[candidate_edge]
        return edge_value > 0.0

    def get_y(self, theta):
        utility_params = dict().fromkeys(['theta'])
        utility_params['theta'] = theta
        utility_params['sparsity'] = 500 * np.sqrt(8 / self.params['size'])
        degrees_df, networks = self.generate_time_series(utility_params,
                                                         suply_network_timeseries=True)
        dummy1 = copy.copy(networks)
        dummy2 = copy.copy(degrees_df.values[:, 0:self.params['feature_length']])
        return copy.deepcopy(dummy1), copy.deepcopy(dummy2)

    def sample_xy(self, n):
        _prior = self.prior
        y_vals = list(range(n))
        theta_vals = list(range(n))
        for ii in range(n):
            y_vals[ii] = dict().fromkeys(('network', 'degrees'))
            theta_vals[ii] = np.random.uniform(self.params['theta_range'][0],
                                               self.params['theta_range'][1])
            y_vals[ii]['network'], y_vals[ii]['degrees'] = self.get_y(theta_vals[ii])
        return [theta_vals, y_vals]

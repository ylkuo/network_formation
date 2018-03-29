# simple test file to test with the outputs of the generative models in with Crunchbase dataset parameters

import settings

from models import *



def plot_observed_time_series(network_timeseries, input_type):
    if input_type == 'clustering':
        all_nodes_clustering = list(
            map(lambda node_pointer: list(map(lambda network: 1.0 * (NX.clustering(network, node_pointer)),
                                              network_timeseries)), network_timeseries[0].nodes()))
        df = pd.DataFrame(np.transpose(all_nodes_clustering))
    elif input_type  == 'transitivity':  # also called global clustering coefficient
        transitivity_timeseies = list(map(lambda network: 1.0 * (NX.transitivity(network)), network_timeseries))
        df = pd.DataFrame(transitivity_timeseies)
    elif input_type == 'avg_clustering':
        all_nodes_clustering = list(
            map(lambda node_pointer: list(map(lambda network: 1.0 * (NX.clustering(network, node_pointer)),
                                              network_timeseries)), network_timeseries[0].nodes()))
        avg_clustering_timeseries = np.sum(all_nodes_clustering, 0) / NX.number_of_nodes(network_timeseries[0])
        df = pd.DataFrame(avg_clustering_timeseries)
    if input_type == 'transitivity':
        df.columns = ['transitivity (global clustering coefficient)']
        df['new links added'] = pd.Series(list(range(len(df))))
        df.plot(x='new links added', y='transitivity (global clustering coefficient)')
        plt.show()
    elif input_type == 'avg_clustering':
        df.columns = ['average clustering']
        df['new links added'] = pd.Series(list(range(len(df))))
        df.plot(x='new links added', y='average clustering')
        plt.show()
    return


if __name__ == '__main__':

    network_timeseries = pickle.load(open('./data/' + 'observed_time_series_sliding_31day_recession_min2.pkl', 'rb'))
    plot_observed_time_series(network_timeseries, settings.params['input_type'])
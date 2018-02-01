import networkx as nx
import sqlite3

from collections import defaultdict
from copy import deepcopy
from operator import itemgetter
from time import strftime, strptime

class CrunchbaseData(object):
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)

    def _get_data_from_table(self, table_name, field):
        cursor = self.conn.cursor()
        results = cursor.execute('SELECT ' + field + ' FROM ' + table_name)
        return results.fetchall()

    def get_investment_network(self):
        results = self._get_data_from_table('investments', '*')
        network = nx.DiGraph()
        for c_link, c_name, c_cat_list, c_country, c_state, c_region, c_city, \
            i_link, i_name, i_country, i_state, i_region, i_city, \
            r_link, r_type, r_code, funded_at, raised_usd in results:
           # create node and edge
           if i_link not in network:
               network.add_node(i_link)
               network.node[i_link]['name'] = i_name
               network.node[i_link]['location'] = \
                   {'country': i_country, 'state': i_state, 'region': i_region, 'city': i_city}
           if c_link not in network:
               network.add_node(c_link)
               network.node[c_link]['name'] = c_name
               network.node[c_link]['location'] = \
                   {'country': c_country, 'state': c_state, 'region': c_region, 'city': c_city}
           if 'category' not in network.node[i_link].keys():
               network.node[i_link]['category'] = defaultdict(int)
           if 'round_type' not in network.node[i_link].keys():
               network.node[i_link]['round_type'] = defaultdict(int)
           network.add_edges_from([(i_link, c_link, {'round': r_link, 'funded_at': funded_at})])
           # build risk profile
           if r_code != '':
               network[i_link][c_link]['round_type'] = r_code
               network.node[i_link]['round_type'][r_code] += 1
           else:
               network[i_link][c_link]['round_type'] = r_type
               network.node[i_link]['round_type'][r_type] += 1
           # build interest profile
           if c_cat_list != '':
               categories = c_cat_list.split('|')
               for cat in categories:
                   network.node[i_link]['category'][cat] += 1
        return network

    def convert_to_coinvest_network(self, in_network):
        out_network = nx.Graph()
        for node in in_network.nodes:
            in_edges = in_network.in_edges(node)
            investors = [edge[0] for edge in in_edges]
            if len(investors) < 2: continue
            # add investor nodes
            for investor in investors:
                if investor in out_network: continue
                out_network.add_node(investor)
                for attr in in_network.node[investor].keys():
                    out_network.node[investor][attr] = \
                        deepcopy(in_network.node[investor][attr])
            # add co-invest edges
            invest_cat = in_network
            for i1 in investors:
                t1 = strptime(in_network[i1][node]['funded_at'], '%Y-%m-%d')
                for i2 in investors:
                    if i1 == i2: continue
                    t2 = strptime(in_network[i1][node]['funded_at'], '%Y-%m-%d')
                    if not out_network.has_edge(i1, i2):
                        out_network.add_edge(i1, i2)
                        out_network[i1][i2]['count'] = 0
                        out_network[i1][i2]['time'] = dict()
                    out_network[i1][i2]['count'] += 1
                    # add first co-investment time of a company
                    if node not in out_network[i1][i2]['time'].keys():
                        if t1 > t2: ts = t1
                        else: ts = t2
                        out_network[i1][i2]['time'][node] = strftime('%Y-%m-%d', ts)
        return out_network


if __name__ == '__main__':
    cb_data = CrunchbaseData('data/crunchbase_2015.db')
    invest_network = cb_data.get_investment_network()
    coinvest_network = cb_data.convert_to_coinvest_network(invest_network)
    # get top co-investors
    print(sorted(coinvest_network.edges(data='count'), \
                 key=itemgetter(2), reverse=True)[:20])
    # get top investors
    print(sorted(invest_network.out_degree(invest_network.nodes), \
                 key=itemgetter(1), reverse=True)[:20])


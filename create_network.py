import networkx as nx
import sqlite3

from collections import defaultdict
from operator import itemgetter

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
               network.node[i_link]['location'] = \
                   {'name': i_name, 'country': i_country, 'state': i_state, \
                    'region': i_region, 'city': i_city}
           if c_link not in network:
               network.add_node(c_link)
               network.node[c_link]['location'] = \
                   {'name': c_name, 'country': c_country, 'state': c_state, \
                    'region': c_region, 'city': c_city}
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
        pass


if __name__ == '__main__':
    cb_data = CrunchbaseData('data/crunchbase_2015.db')
    invest_network = cb_data.get_investment_network()
    # get top investers
    print(sorted(invest_network.out_degree(invest_network.nodes), \
                 key=itemgetter(1),reverse=True)[:20])


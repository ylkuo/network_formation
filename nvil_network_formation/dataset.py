
from GenerativeModel import NetworkFormationGenerativeModel
from torch.utils.data import Dataset

import settings

class NetworkDataset(object):
    def __init__(self, N=300):
        self.len = N
        self.network_formation = NetworkFormationGenerativeModel(settings.gen_model_params)
        [xsamp, ysamp] = self.network_formation.sampleXY(N)
        self.xsamp = xsamp
        self.ysamp = ysamp
        #print('ysamp',ysamp)
        #print('xsamp',xsamp)

    def get_dim(self):
        self.xdim = settings.number_of_classes
        return (self.xdim)

    def get_avg_length_time_series(self):
        sum_of_time_series_lengths = 0
        for i in range(self.len):
            # print(len(self.ysamp[i]['network']))
            sum_of_time_series_lengths += len(self.ysamp[i]['network'])
        return sum_of_time_series_lengths/self.len




class NetworkIterator(object):
    def __init__(self, dataset, batch_size=1):
        self.dataset = dataset
        self.batch_size = batch_size
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.dataset.len:
            raise StopIteration
        else:
            index = self.current
            self.current += self.batch_size
            return (self.dataset.xsamp[index:index+self.batch_size],
                    self.dataset.ysamp[index:index+self.batch_size])



# class Simulations_Dataset(utils_data.Dataset):
#     def __init__(self, n_iters, features, labels):
#         self.ids_list = list(range(len(features)))
#         self.ids_list = random.sample(self.ids_list,n_iters)
#         self.features = features
#         self.labels = labels
#
#     def __getitem__(self, index):
#         feature = self.features[self.ids_list[index]]
#         label = self.labels[self.ids_list[index]]
#         #print(feature)
#         #print(label)
#         return feature, label
#
#     def __len__(self):
#         return len(self.ids_list)
#
# def pad_tensor(vec, pad, dim,pad_with_last_element = True):
#     """
#     args:
#         vec - tensor to pad
#         pad - the size to pad to
#         dim - dimension to pad
#
#     return:
#         a new tensor padded to 'pad' in dimension 'dim'
#     """
#     pad_size = list(vec.shape)
#     pad_size[dim] = pad - vec.size(dim)
#     if pad_with_last_element:
#
#         pad_with = torch.FloatTensor([vec[-1][dim]] * pad_size[0]) # this works with a one-dimensional
#         # time-series may have to change for a multi dimensional time series
#
#         padded_tensor = torch.cat([vec, pad_with], dim=dim)
#
#     else:
#         padded_tensor = torch.cat([vec, torch.zeros(*pad_size)], dim=dim)
#
#     return padded_tensor
#
# class PadCollate:
#     """
#     a variant of callate_fn that pads according to the longest sequence in
#     a batch of sequences, called by DataLoader()
#     """
#
#     def __init__(self, dim=0): #use dim = 0 when batch_first = False
#         """
#         args:
#             dim - the dimension to be padded (dimension of time in sequences)
#         """
#         self.dim = dim
#
#     def pad_collate(self, batch):
#         """
#         args:
#             batch - list of (tensor, label)
#
#         reutrn:
#             xs - a tensor of all examples in 'batch' after padding
#             ys - a LongTensor of all labels in batch
#         """
#         # find longest sequence
#         max_len = max(map(lambda x: x[0].shape[self.dim], batch))
#
#         # pad according to max_len
#         batch = list(map(lambda p:
#                     (pad_tensor(p[0], pad=max_len, dim=self.dim), p[1]), batch))
#         # stack all
#         xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
#         ys = torch.LongTensor(np.array(list(map(lambda x: x[1], batch))))
#         return xs, ys
#
#     def __call__(self, batch):
#         return self.pad_collate(batch)
#
#
#
#
# def gen_torch_data_set(self, dataset_size=1000, filename='dataset.pkl', LOAD=False, SAVE=False):
#     if LOAD:
#         simulation_results = pickle.load(open('./data/' + filename, 'rb'))
#     else:
#         simulation_results = []
#         for iter in range(dataset_size):
#             label_of_data, data = self.gen_torch_sample()
#             # print(data)
#             simulation_results.append((label_of_data, data))
#         if SAVE:
#             pickle.dump(simulation_results, open('./data/' + filename, 'wb'))
#     return simulation_results
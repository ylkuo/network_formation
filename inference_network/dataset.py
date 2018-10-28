import numpy as np
import settings
import torch

from generative_model import GenerativeModel
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class NetworkDataset(Dataset):
    def __init__(self, n_theta, size_per_theta=20, prior_type='normal'):
        self.networks = []
        self.prior_type = prior_type
        for _ in range(n_theta):
            self.sample_theta(size_per_theta)

    def sample_theta(self, size_per_theta):
        if self.prior_type == 'normal':
            theta = np.random.normal(settings.prior_mean, settings.prior_stddev)
        elif self.prior_type == 'uniform':
            theta = np.random.uniform(settings.gen_model_params['theta_range'][0],
                                      settings.gen_model_params['theta_range'][1])
        else:
            assert False, 'Unsupported distribution for data generation.'
        for n in range(size_per_theta):
            model = GenerativeModel(settings.gen_model_params)
            network = dict().fromkeys(('theta', 'degrees'))
            _, network['degrees'] = model.get_y(theta)
            network['degrees'] = torch.tensor(network['degrees']).type(settings.dtype)
            network['theta'] = torch.tensor([theta]).type(settings.dtype)
            self.networks.append(network)

    def __getitem__(self, index):
        return self.networks[index]['degrees'], self.networks[index]['theta']

    def __len__(self):
        return len(self.networks)

'''Utils to pad tensors of variable lengths.
Adapted from https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/7
'''

def pad_tensor(vec, pad, dim):
    """
    args:
        vec - tensor to pad
        pad - the size to pad to
        dim - dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    """
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.size(dim)
    if pad_size[dim] > 0:
        padded_vec = torch.stack([vec[-1] for _ in range(pad_size[dim])])
        return torch.cat([vec, padded_vec], dim=dim)
    else:
        return vec


class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def pad_collate(self, batch):
        """
        args:
            batch - list of (tensor, label)

        reutrn:
            xs - a tensor of all examples in 'batch' after padding
            ys - a LongTensor of all labels in batch
        """
        # find longest sequence
        max_len = max(map(lambda x: x[0].shape[self.dim], batch))
        # pad according to max_len
        batch = [(pad_tensor(x, pad=max_len, dim=self.dim), y) for x, y in batch]
        xs = torch.stack(list(map(lambda x: x[0], batch)), dim=0)
        ys = torch.Tensor(list(map(lambda x: x[1], batch))).unsqueeze(1)
        return xs, ys

    def __call__(self, batch):
        return self.pad_collate(batch)

if __name__ == '__main__':
    # test creatign dataset
    dataset = NetworkDataset(2)
    print('dataset size', len(dataset))
    # test with data loader for batches
    dataloader = DataLoader(dataset, batch_size=10, collate_fn=PadCollate(dim=0))
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched[0].size(), sample_batched[1].size())

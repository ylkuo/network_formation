import torch
import torch.optim as optim
import settings

from dataset import *
from estimator import Estimator
from inference_net import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def train():
    # generate training and validation datasets
    train_data = NetworkDataset(n_theta=settings.n_train,
                                size_per_theta=settings.size_per_theta_train)
    print('Generated training data:', len(train_data))
    val_data = NetworkDataset(n_theta=settings.n_validation, size_per_theta=5)
    print('Generated validation data:', len(val_data))
    # init network
    model = InferenceNetwork()
    if settings.use_gpu:
        model.cuda()
    # training
    writer = SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr=settings.lr,
                           weight_decay=settings.weight_decay)
    for i in range(settings.n_epochs):
        train_loader = DataLoader(train_data, shuffle=True,
                                  batch_size=settings.batch_size,
                                  collate_fn=PadCollate(dim=0))
        val_loader = DataLoader(val_data, batch_size=settings.batch_size,
                                collate_fn=PadCollate(dim=0))
        # train
        model.train()
        n_batches = 0
        batch_loss = 0
        for j, (in_sequences, seq_lengths, thetas, features) in enumerate(train_loader):
            proposal = model.forward(features, in_sequences, seq_lengths)
            loss = proposal.log_prob(thetas)
            loss = -torch.sum(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss
            print('epoch %d, iter %d, loss: %f' % (i, j, float(loss)))
            n_batches += 1
        batch_loss /= n_batches
        writer.add_scalar('loss/train', float(batch_loss), i)
        # validation
        model.eval()
        n_batches = 0
        batch_loss = 0
        for j, (in_sequences, seq_lengths, thetas, features) in enumerate(val_loader):
            proposal = model.forward(features, in_sequences, seq_lengths)
            loss = proposal.log_prob(thetas)
            loss = -torch.sum(loss)
            batch_loss += loss
            n_batches += 1
        batch_loss /= n_batches
        writer.add_scalar('loss/validate', float(batch_loss), i)
        if i > 0 and i % settings.checkpoint_range == 0:
            torch.save(model, settings.model_prefix + 'formation_' + str(i) + '.model')
        torch.save(model, settings.model_prefix + 'formation_' + str(settings.n_epochs) + '.model')
    return model, train_data


if __name__ == '__main__':
    if settings.is_train:
        model, training_data = train()
        torch.save(model, settings.model_prefix + settings.model_name)
    else:
        model = torch.load(settings.model_prefix + settings.model_name)
        if settings.use_gpu:
            model.cuda()
    model.eval()
    estimator = Estimator(model, n_samples=settings.n_eval_theta_samples,
                          estimator_type=settings.estimator_type)
    estimator.get_estimates(settings.n_eval_thetas)

    estimator.get_estimates_on_data(training_data)

import pickle
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import settings

from dataset import *
from estimator import Estimator
from inference_net import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader


def train():
    # generate training and validation datasets
    if settings.load_dataset:
        train_data = pickle.load(open("train.pickle", "rb"))
        val_data = pickle.load(open("val.pickle", "rb"))
    else:
        train_data = NetworkDataset(n_theta=settings.n_train,
                                    size_per_theta=settings.size_per_theta_train)
        print('Generated training data:', len(train_data))
        val_data = NetworkDataset(n_theta=settings.n_validation, size_per_theta=5)
        print('Generated validation data:', len(val_data))
        pickle.dump(train_data, open("train.pickle", "wb" ))
        pickle.dump(val_data, open("val.pickle", "wb" ))
    # init network
    model = InferenceNetwork()
    if settings.use_gpu:
        model.cuda()
    # training
    writer = SummaryWriter()
    optimizer = optim.Adam(model.parameters(), lr=settings.lr,
                           weight_decay=settings.weight_decay)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20,50,80], gamma=1.)
    for i in range(settings.n_epochs):
        scheduler.step()
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
            #loss = -proposal.log_prob(thetas)
            #'''
            loss = 0; n_loss_samples = 50
            for _ in range(n_loss_samples):
                samples = proposal.sample()
                loss += torch.sum(proposal.prob(samples) * F.mse_loss(samples, thetas, reduction='none'))
            loss /= float(n_loss_samples)
            #'''
            #loss = torch.sum(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss += loss
            print('epoch %d, iter %d, loss: %f' % (i, j, float(loss)))
            n_batches += 1
        batch_loss /= n_batches
        writer.add_scalar('loss/train', float(batch_loss), i)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], i)
        # validation
        model.eval()
        n_batches = 0
        batch_loss = 0
        for j, (in_sequences, seq_lengths, thetas, features) in enumerate(val_loader):
            proposal = model.forward(features, in_sequences, seq_lengths)
            samples = proposal.sample()
            for _ in range(n_loss_samples):
                samples = proposal.sample()
                loss += torch.sum(proposal.prob(samples) * F.mse_loss(samples, thetas, reduction='none'))
            loss /= float(n_loss_samples)
            '''
            loss = -proposal.log_prob(thetas)
            loss = torch.sum(loss)
            '''
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
        training_data = pickle.load(open("train.pickle", "rb"))
        model = torch.load(settings.model_prefix + settings.model_name)
        if settings.use_gpu:
            model.cuda()
    model.eval()
    estimator = Estimator(model, n_samples=settings.n_eval_theta_samples,
                          estimator_type=settings.estimator_type)
    estimator.get_estimates(settings.n_eval_thetas)

    estimator.get_estimates_on_data(training_data)

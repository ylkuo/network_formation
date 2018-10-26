import torch
import torch.optim as optim
import settings

from dataset import *
from inference_net import *
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

def train():
    # generate training and validation datasets
    train_data = NetworkDataset(settings.n_train)
    print('Generated training data:', len(train_data))
    val_data = NetworkDataset(settings.n_validation)
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
        train_loader = DataLoader(train_data, batch_size=settings.size_per_theta_train,
                              collate_fn=PadCollate(dim=0))
        val_loader = DataLoader(val_data, batch_size=settings.size_per_theta_train,
                                collate_fn=PadCollate(dim=0))
        # train
        model.train()
        for j, (degrees, thetas) in enumerate(train_loader):
            proposal = model.forward(degrees)
            loss = proposal.log_prob(thetas)
            batch_loss = -torch.sum(loss) / degrees.size()[0]
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            print('epoch %d, iter %d, loss: %f' % (i, j, float(batch_loss)))
        writer.add_scalar('loss/train', float(batch_loss), i)
        # validation
        model.eval()
        for j, (degrees, thetas) in enumerate(val_loader):
            proposal = model.forward(degrees)
            loss = proposal.log_prob(thetas)
            batch_loss = -torch.sum(loss) / degrees.size()[0]
            writer.add_scalar('loss/validate', float(batch_loss), i)
        if i > 0 and i % settings.checkpoint_range == 0:
            torch.save(model, settings.model_prefix + '/formation_' + str(i) + '.model')
        torch.save(model, settings.model_prefix + '/formation_' + str(settings.n_epochs) + '.model')
    return model


if __name__ == '__main__':
    if settings.is_train:
        model = train()
        torch.save(model, settings.model_prefix + settings.model_name)
    else:
        model = torch.load(settings.model_prefix + settings.model_name)
        model.cuda()
    model.eval()

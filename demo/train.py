import torch
import torch.optim as optim
from collections import defaultdict
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import time

from .model import UNet


def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()

    intersection = (pred * target).sum(dim=2).sum(dim=2)

    loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))

    return loss.mean()


def calc_loss(pred, target, metrics, bce_weight=0.5):
    bce = F.binary_cross_entropy_with_logits(pred, target)

    pred = F.sigmoid(pred)
    dice = dice_loss(pred, target)

    loss = bce * bce_weight + dice * (1 - bce_weight)

    metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
    metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
    metrics['loss'] += loss.data.cpu().numpy() * target.size(0)

    return loss


def print_metrics(metrics, epoch_samples, phase):
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))

    print("{}: {}".format(phase, ", ".join(outputs)))


def train_model(model, optimizer, dataloaders, input_data, num_epochs, offset, device):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    model = model.to(device)
    positions = input_data['pos']
    actions = input_data['act']
    h = 2 * offset + 1
    w = 2 * offset + 1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs_, labels_ in enumerate(dataloaders[phase]):
                map_name = inputs_.get_name()
                for current, action in zip(positions[map_name], actions[map_name]):
                    y, x = current
                    inputs = inputs_[:, :, (y-offset):(y+offset+1), (x-offset):(x+offset+1)]

                    action_y = torch.full((1, 1, h, w), action[0])
                    action_x = torch.full((1, 1, h, w), action[1])
                    action_xy = torch.cat((action_y, action_x), dim=0)
                    inputs = torch.cat((inputs, action_xy), dim=1)

                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        loss = calc_loss(outputs, labels, metrics)

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def main():
    num_classes = 2
    num_epochs = 10
    batch_size = 25
    offset = 10

    dataloaders = {'train': DataLoader(train_set, batch_size=batch_size, shuffle=True),
                   'val': DataLoader(val_set, batch_size=batch_size, shuffle=False)}

    model = UNet(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = train_model(model, optimizer, dataloaders, num_epochs, offset, device)


import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np
import wandb

from mcts_dl.utils.dataset import City


def calc_iou(outputs, targets):
    SMOOTH = 1e-6
    outputs = outputs.squeeze(1) > 0.5
    targets = targets.squeeze(1) > 0.5
    intersection = (outputs & targets).float().sum((1, 2))
    union = (outputs | targets).float().sum((1, 2))

    iou = intersection / (union + SMOOTH)

    return iou.sum()


class ModelNetwork(nn.Module):
    def __init__(self, ):
        super(ModelNetwork, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        self.action = nn.Sequential(
            nn.Linear(2, 100),
        )

    def forward(self, inputs, actions):
        return self.model(x)


class Runner:
    def __init__(self, data_set):
        self.model = ModelNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 4
        self.num_epochs = 100
        self.data_loaders = {'train': DataLoader(data_set['train'], batch_size=self.batch_size, shuffle=True),
                             'val': DataLoader(data_set['val'], batch_size=self.batch_size, shuffle=False)}
        self.offset = 21
        self.num_steps = 1000
        self.loss_func = nn.BCELoss()

    # check random seed
    def sample(self, image_map):
        y0, x0 = np.random.randint(self.offset + 1, 256 - self.offset - 1, size=(1, 2))[0]
        inputs = image_map[:, :, (y0 - self.offset):(y0 + self.offset + 1), (x0 - self.offset):(x0 + self.offset + 1)]

        targets = torch.zeros_like(inputs)
        actions = torch.zeros(inputs.shape[0])
        for i in range(inputs.shape[0]):
            dy, dx = np.random.randint(-1, 2, size=(1, 2))[0]
            y = y0 + dy
            x = x0 + dx
            targets[i] = image_map[i, :, (y - self.offset):(y + self.offset + 1), (x - self.offset):(x + self.offset + 1)]
            actions[i] = torch.tensor([dy, dx], dtype=torch.int8)

        return inputs, targets, actions

    def train(self):
        self.model.train()

        epoch_loss = 0.0
        epoch_iou = 0.0
        for batch in self.data_loaders['train']:
            image_map = batch['image']
            for step in range(self.num_steps):
                inputs, targets, actions = self.sample(image_map)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                actions = actions.to(self.device)

                outputs = self.model(inputs, actions)

                loss = self.loss_func(outputs, targets)
                epoch_loss += loss.cpu().detach()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                iou = calc_iou(outputs.cpu().detach(), targets.cpu().detach())
                epoch_iou += iou

        epoch_loss = epoch_loss / (len(self.data_loaders['train']) * self.num_steps * self.batch_size)
        epoch_iou = epoch_iou / (len(self.data_loaders['train']) * self.num_steps * self.batch_size)

        return epoch_loss, epoch_iou

    def eval(self):
        self.model.eval()

        epoch_loss = 0.0
        epoch_iou = 0.0
        for batch in self.data_loaders['val']:
            image_map = batch['image']
            for step in range(self.num_steps):
                inputs, targets, actions = self.sample(image_map)

                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                actions = actions.to(self.device)

                outputs = self.model(inputs, actions)

                loss = self.loss_func(outputs, targets)
                epoch_loss += loss.cpu().detach()

                iou = calc_iou(outputs.cpu().detach(), targets.cpu().detach())
                epoch_iou += iou

        epoch_loss = epoch_loss / (len(self.data_loaders['val']) * self.num_steps * self.batch_size)
        epoch_iou = epoch_iou / (len(self.data_loaders['val']) * self.num_steps * self.batch_size)

        return epoch_loss, epoch_iou

    def run(self):
        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            train_loss, train_iou = self.train()
            val_loss, val_iou = self.eval()

            logs = {'train_loss': train_loss,
                    'train_iou': train_iou,
                    'val_loss': val_loss,
                    'val_iou': val_iou}
            wandb.log(logs)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.load_state_dict(best_model_wts)
        return self.model


if __name__ == '__main__':
    wandb.init(project="mcts-dl")
    city_ds = City(map_root="../../data/street-map")
    data_set = {'train': city_ds,
                'val': city_ds}

    runner = Runner(data_set)
    best_model = runner.run()

    torch.save(best_model.state_dict(), "models/best_model.pth")


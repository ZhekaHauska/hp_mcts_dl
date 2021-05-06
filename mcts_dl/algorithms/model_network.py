import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np
import wandb

from mcts_dl.utils.dataset import City
from mcts_dl.utils.iou import calc_iou


class ModelNetwork(nn.Module):
    def __init__(self, window_size):
        super(ModelNetwork, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.action = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, window_size*window_size),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(8+1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, inputs, actions):
        inputs_ = self.input(inputs)
        actions_ = self.action(actions)
        actions_ = actions_.view(inputs.shape)

        outputs = torch.cat((inputs_, actions_), dim=1)
        outputs = self.output(outputs)

        return outputs


class Runner:
    def __init__(self, config):
        self.config = config

        self.map_size = config['map_size']
        train_ds = City(map_root="../../data/train", map_size=self.map_size)
        val_ds = City(map_root="../../data/val", map_size=self.map_size)
        test_ds = City(map_root="../../data/test", map_size=self.map_size)

        data_set = {'train': train_ds,
                    'val': val_ds,
                    'test': test_ds}

        self.offset = config['offset']

        self.window_size = 2 * self.offset + 1
        self.model = ModelNetwork(self.window_size)

        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = config['num_epochs']

        self.data_loaders = {'train': DataLoader(data_set['train'], batch_size=6, shuffle=True),
                             'val': DataLoader(data_set['val'], batch_size=2, shuffle=True),
                             'test': DataLoader(data_set['test'], batch_size=2, shuffle=True)}

        self.num_steps = config['num_steps']
        self.threshold = config['threshold']
        self.checkpoints_dir = f"{config['checkpoints_dir']}/offset_{config['offset']}_num_steps_{config['num_steps']}"

    def sample(self, image_map):
        y0, x0 = np.random.randint(self.offset + 1, self.map_size - self.offset - 1, size=(1, 2))[0]
        inputs = image_map[:, :, (y0 - self.offset):(y0 + self.offset + 1), (x0 - self.offset):(x0 + self.offset + 1)]

        targets = torch.zeros_like(inputs)
        actions = torch.zeros((inputs.shape[0], 2))

        for i in range(inputs.shape[0]):
            dy, dx = np.random.randint(-1, 2, size=(1, 2))[0]
            y = y0 + dy
            x = x0 + dx
            targets[i] = image_map[i, :, (y - self.offset):(y + self.offset + 1), (x - self.offset):(x + self.offset + 1)]
            actions[i][0] = dy
            actions[i][1] = dx

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

                iou = calc_iou(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                epoch_iou += iou

        epoch_loss = epoch_loss / (len(self.data_loaders['train']) * self.num_steps)
        epoch_iou = epoch_iou / (len(self.data_loaders['train']) * self.num_steps)

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

                iou = calc_iou(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                epoch_iou += iou

        epoch_loss = epoch_loss / (len(self.data_loaders['val']) * self.num_steps)
        epoch_iou = epoch_iou / (len(self.data_loaders['val']) * self.num_steps)

        log_window = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)

        output = outputs[0].cpu().detach().squeeze() > self.threshold
        target = targets[0].cpu().detach().squeeze() > self.threshold

        intersection = (output & target)
        log_window[output] = [255, 0, 0]
        log_window[target] = [0, 255, 0]
        log_window[intersection] = [255, 255, 255]

        return epoch_loss, epoch_iou, log_window

    def pred(self, inputs):
        model_path = f"{self.checkpoints_dir}/best_model.pth"
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        outputs = self.model(inputs)

        return outputs

    def run(self, log=True):
        np.random.seed(0)
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        if log:
            wandb.init(project=self.config['project_name'], config=self.config)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            train_loss, train_iou = self.train()
            val_loss, val_iou, log_window = self.eval()

            logs = {'train_loss': train_loss,
                    'train_iou': train_iou,
                    'val_loss': val_loss,
                    'val_iou': val_iou}
            wandb.log(logs)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

                wandb.log({f"epoch = {epoch}": [wandb.Image(log_window, caption='red=pred, '
                                                                                'green=true, '
                                                                                'white=intersection')]})
                # for im, c in zip([inputs[0], outputs[0], targets[0]], ['input', 'output', 'target'])]})

            torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/epoch_{epoch:05d}.pth")

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/best_model.pth")


if __name__ == '__main__':
    import yaml

    with open('../../configs/model_network/default.yaml', 'r') as file:
        config = yaml.load(file, yaml.Loader)

    runner = Runner(config)
    runner.run()




import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np
import wandb

from mcts_dl.utils.dataset import City
from mcts_dl.utils.metrics import calc_iou, calc_acc, calc_f1
from mcts_dl.environment.gridworld_pomdp import calculate_next_vector, is_move_possible


class ModelNetworkWindow(nn.Module):
    def __init__(self, window_size):
        super(ModelNetworkWindow, self).__init__()
        self.input = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, kernel_size=3, padding=1),
            nn.ReLU()
        )

        self.action = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, window_size * window_size),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Conv2d(8 + 1, 4, kernel_size=3, padding=1),
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


class ModelNetworkBorder(nn.Module):
    def __init__(self, window_size=21, map_size=256):
        super(ModelNetworkBorder, self).__init__()
        self.window_size = window_size
        self.map_size = map_size

        self.input = nn.Sequential(
            nn.Conv2d(1, 2, 3),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Conv2d(2, 4, 3),
            nn.BatchNorm2d(4),
            nn.ReLU()
        )

        self.output = nn.Sequential(
            nn.Linear(1156, 4 * window_size + 4),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        inputs = self.input(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        outputs = self.output(inputs)

        return outputs

    def make_window(self, inputs, outputs, actions):
        start = 0
        end = self.window_size + 2
        up = outputs[:, start:end]

        start = end
        end = start + self.window_size
        right = outputs[:, start:end]

        start = end
        end = start + self.window_size + 2
        down = outputs[:, start:end]

        start = end
        end = start + self.window_size
        left = outputs[:, start:end]

        result = torch.zeros((inputs.shape[0], 1, self.window_size + 2, self.window_size + 2))
        result[:, :, 0, :] = up.unsqueeze(1)
        result[:, :, 1:-1, -1] = right.unsqueeze(1)
        result[:, :, -1, :] = down.unsqueeze(1)
        result[:, :, 1:-1, 0] = left.unsqueeze(1)
        result[:, :, 1:-1, 1:-1] = inputs.cpu().detach()

        outputs = torch.zeros_like(inputs)
        for idx, action in enumerate(actions):
            dy = int(actions[idx, 0].item())
            dx = int(actions[idx, 1].item())

            y0 = self.window_size//2 + 1
            x0 = self.window_size//2 + 1

            y = y0 + dy
            x = x0 + dx

            outputs[idx] = result[idx, :, (y - self.window_size//2):(y + self.window_size//2 + 1), (x - self.window_size//2):(x + self.window_size//2 + 1)]

        threshold = 0.5
        outputs = outputs.cpu().detach().squeeze() > threshold

        return outputs.float()

    def predict(self, window, action, device):
        window = torch.from_numpy(window).float()

        inputs = window.reshape(1, 1, window.shape[0], window.shape[0])
        inputs = inputs.to(device)

        outputs = self.forward(inputs)
        actions = action.unsqueeze(0)
        next_window = self.make_window(inputs, outputs, actions)

        return next_window.numpy()

    def run(self, observation, action, device):
        window = observation[0]
        vector = observation[1]
        displacement = action
        max_length = np.sqrt(2) * self.map_size
        next_vector = calculate_next_vector(vector, displacement, max_length)

        next_window = self.predict(window, action, device)

        return (next_window, next_vector)


class Runner:
    def __init__(self, config):
        self.config = config
        self.mode = config['mode']
        self.map_size = config['map_size']

        train_ds = City(map_root="../../data/train", map_size=self.map_size)
        val_ds = City(map_root="../../data/val", map_size=self.map_size)

        data_set = {'train': train_ds,
                    'val': val_ds}

        self.offset = config['offset']

        self.window_size = 2 * self.offset + 1

        models = {'window': ModelNetworkWindow,
                  'border': ModelNetworkBorder}
        self.model = models[self.mode](self.window_size)

        self.loss_func = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_epochs = config['num_epochs']

        self.data_loaders = {'train': DataLoader(data_set['train'], batch_size=6, shuffle=True),
                             'val': DataLoader(data_set['val'], batch_size=4, shuffle=True)}

        self.num_steps = config['num_steps']
        self.threshold = config['threshold']
        self.checkpoints_dir = config['checkpoints_dir']

    def sample_window(self, image_map):
        y0, x0 = np.random.randint(self.offset + 1, self.map_size - self.offset - 1, size=(1, 2))[0]
        # is free place
        while image_map[:, :, y0-1:y0+2, x0-1:x0+1].sum() != 0:
            y0, x0 = np.random.randint(self.offset + 1, self.map_size - self.offset - 1, size=(1, 2))[0]

        inputs = image_map[:, :, (y0 - self.offset):(y0 + self.offset + 1), (x0 - self.offset):(x0 + self.offset + 1)]

        targets = torch.zeros_like(inputs)
        actions = torch.zeros((inputs.shape[0], 2))

        for i in range(inputs.shape[0]):
            dy, dx = np.random.randint(-1, 2, size=(1, 2))[0]
            # obs_window = inputs[i][0]
            # displacement = (dy, dx)
            while dy == 0 and dx == 0:  # or not is_move_possible(obs_window, displacement):
                dy, dx = np.random.randint(-1, 2, size=(1, 2))[0]
                # displacement = (dy, dx)

            y = y0 + dy
            x = x0 + dx

            targets[i] = image_map[i, :, (y - self.offset):(y + self.offset + 1), (x - self.offset):(x + self.offset + 1)]
            actions[i][0] = dy
            actions[i][1] = dx

        return inputs, targets, actions

    def sample_border(self, image_map):
        y0, x0 = np.random.randint(self.offset + 1, self.map_size - self.offset - 1, size=(1, 2))[0]
        # is free place
        while image_map[:, :, y0, x0].sum() != 0:
            y0, x0 = np.random.randint(self.offset + 1, self.map_size - self.offset - 1, size=(1, 2))[0]

        inputs = image_map[:, :, (y0 - self.offset):(y0 + self.offset + 1), (x0 - self.offset):(x0 + self.offset + 1)]
        targets = torch.zeros((inputs.shape[0], 4 * self.window_size + 4))
        actions = torch.zeros((inputs.shape[0], 2))
        target_windows = torch.zeros_like(inputs)
        for i in range(inputs.shape[0]):
            dy, dx = np.random.randint(-1, 2, size=(1, 2))[0]
            # obs_window = inputs[i][0]
            # displacement = (dy, dx)
            while dy == 0 and dx == 0: # or not is_move_possible(obs_window, displacement):
                dy, dx = np.random.randint(-1, 2, size=(1, 2))[0]
                # displacement = (dy, dx)

            y = y0
            x = x0

            up = image_map[i, :, (y - self.offset - 1), (x - self.offset - 1):(x + self.offset + 2)]
            down = image_map[i, :, (y + self.offset + 1), (x - self.offset - 1):(x + self.offset + 2)]

            right = image_map[i, :, (y - self.offset):(y + self.offset + 1), (x + self.offset + 1)]
            left = image_map[i, :, (y - self.offset):(y + self.offset + 1), (x - self.offset - 1)]

            targets[i] = torch.cat((up, right, down, left), dim=-1)

            y = y0 + dy
            x = x0 + dx
            target_windows[i] = image_map[i, :, (y - self.offset):(y + self.offset + 1),
                                (x - self.offset):(x + self.offset + 1)]

            actions[i][0] = dy
            actions[i][1] = dx

        return inputs, targets, actions, target_windows

    def train(self):
        self.model.train()
        phase = 'train'

        epoch_metrics = dict()
        epoch_metrics['loss'] = 0.0

        if self.mode == 'border':
            epoch_metrics['acc'] = 0.0
            epoch_metrics['f1'] = 0.0
        else:
            epoch_metrics['iou'] = 0.0

        for batch in self.data_loaders[phase]:
            image_map = batch['image']
            for step in range(self.num_steps):
                if self.mode == 'border':
                    inputs, targets, actions, _ = self.sample_border(image_map)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    outputs = self.model(inputs)
                else:
                    inputs, targets, actions = self.sample_window(image_map)
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    actions = actions.to(self.device)

                    outputs = self.model(inputs, actions)

                # window = inputs[0][0].cpu().numpy()
                # action = actions[0]
                # next_window = self.model.predict(window, action, self.device)

                loss = self.loss_func(outputs, targets)
                epoch_metrics['loss'] += loss.cpu().detach()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if self.mode == 'border':
                    acc = calc_acc(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                    epoch_metrics['acc'] += acc

                    f1 = calc_f1(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                    epoch_metrics['f1'] += f1
                else:
                    iou = calc_iou(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                    epoch_metrics['iou'] += iou

        for m in epoch_metrics:
            epoch_metrics[m] = epoch_metrics[m] / (len(self.data_loaders[phase]) * self.num_steps)

        return epoch_metrics

    def make_log_window(self, inputs, outputs, actions, target_windows):
        log_window = np.zeros((self.window_size, self.window_size, 3), dtype=np.uint8)
        idx = 0
        dy = int(actions[idx, 0].item())
        dx = int(actions[idx, 1].item())

        if self.mode == 'border':
            start = 0
            end = self.window_size + 2
            up = outputs[:, start:end]

            start = end
            end = start + self.window_size
            right = outputs[:, start:end]

            start = end
            end = start + self.window_size + 2
            down = outputs[:, start:end]

            start = end
            end = start + self.window_size
            left = outputs[:, start:end]

            result = torch.zeros((inputs.shape[0], 1, self.window_size + 2, self.window_size + 2))
            result[:, :, 0, :] = up.unsqueeze(1)
            result[:, :, 1:-1, -1] = right.unsqueeze(1)
            result[:, :, -1, :] = down.unsqueeze(1)
            result[:, :, 1:-1, 0] = left.unsqueeze(1)
            result[:, :, 1:-1, 1:-1] = inputs.cpu().detach()

            y0 = self.offset + 1
            x0 = self.offset + 1

            y = y0 + dy
            x = x0 + dx

            output = result[idx, :, (y - self.offset):(y + self.offset + 1), (x - self.offset):(x + self.offset + 1)]
            output = output.cpu().detach().squeeze() > self.threshold
            target = target_windows[idx].cpu().detach().squeeze() > self.threshold
        else:
            output = outputs[idx].cpu().detach().squeeze() > self.threshold
            target = target_windows[idx].cpu().detach().squeeze() > self.threshold

        intersection = (output & target)
        log_window[output] = [255, 0, 0]
        log_window[target] = [0, 255, 0]
        log_window[intersection] = [255, 255, 255]

        log = dict()
        log['image'] = log_window
        log['dy'] = dy
        log['dx'] = dx

        return log

    def eval(self):
        self.model.eval()
        phase = 'val'
        epoch_metrics = dict()
        epoch_metrics['loss'] = 0.0

        if self.mode == 'border':
            epoch_metrics['acc'] = 0.0
            epoch_metrics['f1'] = 0.0
        else:
            epoch_metrics['iou'] = 0.0

        with torch.no_grad():
            for batch in self.data_loaders[phase]:
                image_map = batch['image']
                for step in range(self.num_steps):
                    if self.mode == 'border':
                        inputs, targets, actions, target_windows = self.sample_border(image_map)
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        outputs = self.model(inputs)
                    else:
                        inputs, targets, actions = self.sample_window(image_map)
                        target_windows = targets

                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device)

                        actions = actions.to(self.device)

                        outputs = self.model(inputs, actions)

                    loss = self.loss_func(outputs, targets)
                    epoch_metrics['loss'] += loss.cpu().detach()

                    if self.mode == 'border':
                        acc = calc_acc(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                        epoch_metrics['acc'] += acc

                        f1 = calc_f1(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                        epoch_metrics['f1'] += f1
                    else:
                        iou = calc_iou(outputs.cpu().detach(), targets.cpu().detach(), threshold=self.threshold)
                        epoch_metrics['iou'] += iou

        for m in epoch_metrics:
            epoch_metrics[m] = epoch_metrics[m] / (len(self.data_loaders[phase]) * self.num_steps)

        log = self.make_log_window(inputs, outputs, actions, target_windows)

        return epoch_metrics, log

    def run(self):
        wandb.init(project=self.config['project_name'], config=self.config)

        np.random.seed(0)
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = 1e10

        self.model = self.model.to(self.device)

        for epoch in range(self.num_epochs):
            train_metrics = self.train()
            val_metrics, log = self.eval()

            logs = {'train': train_metrics,
                    'val': val_metrics}
            wandb.log(logs, step=epoch)

            val_loss = val_metrics['loss']
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())

                wandb.log({f"epoch = {epoch}": [wandb.Image(log['image'], caption=f"({log['dy']},{log['dx']})")]})

                # for im, c in zip([inputs[0], outputs[0], targets[0]], ['input', 'output', 'target'])]})

            # torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/epoch_{epoch:04d}.pth")

        self.model.load_state_dict(best_model_wts)
        torch.save(self.model.state_dict(), f"{self.checkpoints_dir}/model_net_{self.mode}_{self.window_size}.pth")


if __name__ == '__main__':
    import yaml

    with open('../../configs/model_network/default.yaml', 'r') as file:
        config = yaml.load(file, yaml.Loader)

    runner = Runner(config)
    runner.run()

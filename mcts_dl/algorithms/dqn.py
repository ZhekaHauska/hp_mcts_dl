import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

import imageio

from mcts_dl.environment.gridworld_pomdp import GridWorld


Transition = namedtuple('Transition',
                        ('state', 'vector', 'action', 'next_state', 'next_vector', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, input_channels, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, padding=2, kernel_size=5, stride=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, padding=0, kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, padding=0, kernel_size=2, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv_out_size(w, padding, kernel, stride):
            return int((w + 2 * padding - (kernel - 1) - 1) / stride + 1)

        out_h = conv_out_size(conv_out_size(conv_out_size(h, 2, 5, 3), 0, 3, 2), 0, 2, 1)
        out_w = conv_out_size(conv_out_size(conv_out_size(w, 2, 5, 3), 0, 3, 2), 0, 2, 1)

        self.fc = nn.Linear(32 * out_h * out_w + 2, 64)
        self.head = nn.Linear(64, outputs)

    def forward(self, x, v):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.fc(torch.cat([x.view(x.size(0), -1), v], dim=-1)))
        return self.head(x)


class DQNBase(nn.Module):
    def __init__(self, input_channels, h, w, outputs):
        super(DQNBase, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(16, 32)
        self.head = nn.Linear(32, outputs)

    def forward(self, x, v):
        x = F.relu(self.fc1(v))
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        return self.head(x)


class DQNAgent:
    def __init__(self, batch_size=None, gamma=None, eps_start=None, eps_end=None, eps_decay=None, target_update=None,
                 window=None, n_actions=None, n_layers=None, capacity=None, learning_rate=None):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update = target_update
        self.window = window
        self.n_actions = n_actions
        self.n_input_layers = n_layers
        self.rows = window
        self.cols = window
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.capacity = capacity
        self.learning_rate = learning_rate

        self.policy_net = DQNBase(self.n_input_layers, self.rows, self.cols, self.n_actions).to(self.device)
        self.target_net = DQNBase(self.n_input_layers, self.rows, self.cols, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.steps_done = 0
        self.memory = ReplayMemory(self.capacity)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)

    def make_action(self, state):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(*state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        non_final_next_vectors = torch.cat([v for v in batch.next_vector
                                           if v is not None])
        state_batch = torch.cat(batch.state)
        vector_batch = torch.cat(batch.vector)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch, vector_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states, non_final_next_vectors).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


class DQNAgentRunner:
    def __init__(self, config):
        self.config = config
        agent_conf = config['agent']
        env_conf = config['environment']

        agent_conf['window'] = env_conf['window_size']

        self.num_episodes = config['episodes']
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = DQNAgent(**agent_conf)
        self.env = GridWorld(**env_conf)

        self.reward_history = list()

    def run(self, learn=True, log=True, log_video_every=100):
        if log:
            run = wandb.init(project=self.config['project_name'], config=self.config)
            world, rmap = self.env.render()
            wandb.log({'task': plt.imshow(world)}, step=0)
            wandb.log({'rmap': plt.imshow(rmap)}, step=0)

        animation = False

        for i_episode in tqdm(range(self.num_episodes), disable=not log):
            # Initialize the environment and state
            self.env.reset()
            obs, reward, is_done = self.env.observe()
            last_state = obs[0]
            current_state = last_state

            state = torch.from_numpy(np.concatenate((current_state - last_state, current_state), axis=0)[None])
            state = state.float().to(device=self.device)
            vector = torch.from_numpy(obs[1][None]).float().to(device=self.device)

            total_reward = 0
            if (i_episode % log_video_every == 0) and log:
                animation = True

            for t in count():
                # Select and perform an action
                action = self.agent.make_action((state, vector))
                self.env.act(action)
                # Observe new state
                obs, reward, is_done = self.env.observe()
                last_state = current_state
                current_state = obs[0]

                total_reward += reward
                reward = torch.tensor([reward], device=self.device).float()

                if not is_done:
                    next_state = torch.from_numpy(np.concatenate((last_state, current_state), axis=0)[None])
                    next_state = next_state.float().to(device=self.device)
                    next_vector = torch.from_numpy(obs[1][None]).float().to(device=self.device)
                else:
                    next_state = None
                    next_vector = None

                    # Store the transition in memory
                self.agent.memory.push(state, vector, action, next_state, next_vector, reward)

                # Move to the next state
                state = next_state
                vector = next_vector

                # Perform one step of the optimization (on the policy network)
                if learn:
                    self.agent.optimize_model()

                if animation:
                    world, rmap = self.env.render()
                    plt.imsave(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{t}.png', world)

                if is_done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if (i_episode % self.agent.target_update == 0) and learn:
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            if log:
                wandb.log({'reward': total_reward, 'duration': t + 1}, step=i_episode)

            if animation:
                animation = False
                with imageio.get_writer(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', mode='I', fps=2) as writer:
                    for i in range(t):
                        image = imageio.imread(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{i}.png')
                        writer.append_data(image)
                wandb.log({f'animation': wandb.Video(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', fps=2,
                                                       format='gif')}, step=i_episode)
            self.reward_history.append(total_reward)


if __name__ == '__main__':
    import yaml
    with open('../../configs/dqn/default.yaml', 'r') as file:
        config = yaml.load(file, yaml.Loader)

    runner = DQNAgentRunner(config)
    runner.run(learn=True, log=True, log_video_every=20)

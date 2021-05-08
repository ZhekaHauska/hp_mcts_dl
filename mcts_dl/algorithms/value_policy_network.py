import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import math
from collections import namedtuple
from itertools import count
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt

import imageio
import random
import os

from mcts_dl.environment.gridworld_pomdp import GridWorld
from mcts_dl.algorithms.dqn import ReplayMemory

Example = namedtuple('Example',
                     ('state', 'vector', 'action_probs', 'value'))


class ValuePolicyNetwork(nn.Module):
    def __init__(self, input_channels, h, w, input_vector_size, n_actions):
        super(ValuePolicyNetwork, self).__init__()
        # window
        self.conv1 = nn.Conv2d(input_channels, 16, padding=0, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, padding=0, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, padding=0, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm2d(32)

        def conv_out_size(w, padding, kernel, stride):
            return int((w + 2 * padding - (kernel - 1) - 1) / stride + 1)

        out_h = conv_out_size(conv_out_size(conv_out_size(h, 0, 3, 1), 0, 3, 1), 0, 3, 1)
        out_w = conv_out_size(conv_out_size(conv_out_size(w, 0, 3, 1), 0, 3, 1), 0, 3, 1)

        self.fc1 = nn.Linear(32 * out_h * out_w, 64)
        self.bn4 = nn.BatchNorm1d(64)

        # vector
        self.fc_vec1 = nn.Linear(input_vector_size, 32)
        self.bn_vec1 = nn.BatchNorm1d(32)
        self.fc_vec2 = nn.Linear(32, 64)
        self.bn_vec2 = nn.BatchNorm1d(64)

        # value
        self.fc_value = nn.Linear(128, 64)
        self.bn_value = nn.BatchNorm1d(64)
        self.value_head = nn.Linear(64, 1)
        # policy
        self.fc_policy = nn.Linear(128, 64)
        self.bn_policy = nn.BatchNorm1d(64)
        self.policy_head = nn.Linear(64, n_actions)

    def forward(self, x, v):
        # window
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        x = self.bn4(x)
        # vector
        v = F.relu(self.fc_vec1(v))
        v = self.bn_vec1(v)
        v = F.relu(self.fc_vec2(v))
        v = self.bn_vec2(v)
        # both
        y = torch.cat([x, v], dim=-1)
        # value
        value = F.relu(self.fc_value(y))
        value = self.bn_value(value)
        value = self.value_head(value)
        # policy
        policy = F.relu(self.fc_policy(y))
        policy = self.bn_policy(policy)
        policy = self.policy_head(policy)
        return value, policy


class VPAgent:
    def __init__(self, batch_size=None, gamma=None, eps_start=None, eps_end=None, eps_decay=None,
                 window=None, n_actions=None, n_layers=None, capacity=None, learning_rate=None, input_vector_size=None):
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.window = window
        self.n_actions = n_actions
        self.n_input_layers = n_layers
        self.rows = window
        self.cols = window
        self.input_vector_size = input_vector_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.capacity = capacity
        self.learning_rate = learning_rate

        self.model = ValuePolicyNetwork(self.n_input_layers, self.rows, self.cols, self.input_vector_size, self.n_actions)
        self.model.to(self.device)

        self.steps_done = 0
        self.memory = ReplayMemory(self.capacity, obj=Example)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.learning_rate)
        self.n_optimizations = 0
        self.steps_since_last_opt = 0

    def make_action(self, state, evaluation=False):
        noise = self.eps_end + (self.eps_start - self.eps_end) * \
                math.exp(-1. * self.steps_done / self.eps_decay)
        self.model.eval()
        scores = self.model(*state)[1]
        if evaluation:
            action = torch.argmax(scores[0]).cpu().detach().numpy()
        else:
            probs = torch.softmax(scores, dim=-1)
            probs = probs + noise
            probs /= probs.sum()
            probs = probs.cpu().detach().numpy()
            action = np.random.choice(np.arange(probs.size), p=probs.flatten())

            self.steps_done += 1
            self.steps_since_last_opt += 1
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return 0, 0
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Example(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        vector_batch = torch.cat(batch.vector)
        exp_probs_batch = torch.cat(batch.action_probs)
        exp_value_batch = torch.cat(batch.value)

        self.model.train()
        value_batch, probs_batch = self.model(state_batch, vector_batch)
        probs_batch = F.log_softmax(probs_batch, dim=-1)
        value_loss = F.smooth_l1_loss(value_batch.squeeze(), exp_value_batch)
        policy_loss = F.kl_div(probs_batch, exp_probs_batch, reduction='sum')
        # Optimize the model
        self.optimizer.zero_grad()
        (0.5*value_loss + policy_loss).backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.n_optimizations += 1
        self.steps_since_last_opt = 0
        return value_loss.item(), policy_loss.item()


class VPAgentCurriculum:
    def __init__(self, config):
        self.config = config
        agent_conf = config['agent']
        self.env_conf = config['environment']

        agent_conf['window'] = self.env_conf['window_size']

        self.max_episodes = config['max_episodes']  # for any level
        self.evaluate_every_episodes = config['eval_period_episodes']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = VPAgent(**agent_conf)

        self.checkpoint_name = config['load_checkpoint']
        self.checkpoint_path = config['checkpoint_path']
        if self.checkpoint_name is not None:
            checkpoint = torch.load(os.path.join(self.checkpoint_path, self.checkpoint_name))
            self.agent.model.load_state_dict(checkpoint['state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        self.start_level = config['start_level']
        self.end_level = config['end_level']
        self.current_level = self.start_level
        self.success_rate_next_level = config['success_rate_next_level']
        self.change_level_every = config['change_level_every']
        self.levels_at_once = config['levels_at_once']
        self.optimize_every_step = config['optimize_every_step']
        self.full_evaluation_period = config['full_evaluation_period']

        self.map_names = config['maps']
        if self.map_names is None:
            self.map_names = filter(lambda x: x.split('.')[-1] == 'map', os.listdir(self.env_conf['path']))
            self.map_names = [x.split('.')[0] for x in self.map_names]

        # 30 maps 30x920 tasks
        # division 6 2 2
        self.envs = self.load_envs()

    def load_envs(self, n_maps=None, eval=False, levels=None):
        indices = list(range(len(self.map_names)))
        if n_maps is not None:
            indices = random.sample(indices, n_maps)
        if levels is None:
            levels = [self.current_level]

        envs = list()
        for level in levels:
            if eval:
                start_task = level * 10 + 6
                end_task = level * 10 + 8
            else:
                start_task = level * 10
                end_task = level * 10 + 6

            for i in indices:
                for task in range(start_task, end_task):
                    self.env_conf['task'] = task
                    self.env_conf['map_name'] = self.map_names[i]
                    envs.append(GridWorld(**self.env_conf))
        return envs

    def evaluate(self, levels=None, log_animation=False, log_animation_every=10, counter_start=0):
        envs = self.load_envs(eval=True, levels=levels)
        completed = np.zeros(len(envs))
        path_difference = np.zeros(len(envs))
        i = counter_start
        for j, env in enumerate(envs):
            is_success, duration, task_length = self.run_episode(env, i,
                                                                 log_metrics=False,
                                                                 eval=True,
                                                                 log_animation=(log_animation and ((i % log_animation_every) == 0)))
            completed[j] = int(is_success)
            if task_length != 0:
                path_difference[j] = (env.path_length - task_length) / task_length

            i += 1

        return i, completed.mean(), path_difference[completed == 1].mean()

    def run_episode(self, env, i_episode, log_metrics=True, log_animation=False, eval=False):
        env.reset()

        for t in count():
            obs, reward, is_done = env.observe()

            if log_animation:
                world, rmap = env.render()
                plt.imsave(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{t}.png', world)

            if is_done:
                break

            action_probs = env.get_action_probs()
            action_probs = torch.from_numpy(action_probs[None]).float().to(device=self.device)
            value = env.get_value()
            value = torch.from_numpy(value[None]).float().to(device=self.device)
            window, vector = obs
            window = torch.from_numpy(window[None])
            window = window.float().to(device=self.device)
            vector = torch.from_numpy(vector[None]).float().to(device=self.device)
            # Select and perform an action
            action = self.agent.make_action((window, vector), evaluation=eval)
            env.act(action)

            # Store state in memory
            if not eval:
                self.agent.memory.push(window, vector, action_probs, value)

        if log_metrics:
            wandb.log({'duration': t}, step=i_episode)

        if log_animation:
            with imageio.get_writer(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', mode='I', fps=3) as writer:
                for i in range(t):
                    image = imageio.imread(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{i}.png')
                    writer.append_data(image)
            wandb.log({f'animation': wandb.Video(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', fps=3,
                                                 format='gif')}, step=i_episode)

        return env.is_success, t, env.task_length

    def run_curriculum(self, log=True, log_video_every=100):
        if log:
            run = wandb.init(project=self.config['project_name'], config=self.config)
            wandb.log({'level': self.current_level}, step=0)

        episode = 0
        episode_level = 0
        evaluations = 0
        while True:
            # choose map and task
            env = random.choice(self.envs)
            self.run_episode(env,
                             log_metrics=log,
                             log_animation=((episode % log_video_every) == 0) and log,
                             i_episode=episode)
            if self.agent.steps_since_last_opt > self.optimize_every_step:
                value_loss, policy_loss = self.agent.optimize_model()
                if log:
                    wandb.log({'value_loss': value_loss, 'policy_loss': policy_loss,
                               'n_optimization': self.agent.n_optimizations}, step=episode)

            if ((episode % self.evaluate_every_episodes) == 0) or (episode == self.max_episodes):
                _, success_rate, path_difference = self.evaluate()
                if log:
                    wandb.log({'eval_success_rate': success_rate,
                               'path_difference': path_difference}, step=episode)
                if success_rate >= self.success_rate_next_level:
                    episode_level = 0
                    self.current_level += 1
                    if self.current_level > self.end_level:
                        self.current_level = self.start_level
                    self.envs = self.load_envs()
                    if log:
                        wandb.log({'level': self.current_level}, step=episode)

                torch.save({
                    'episode': episode,
                    'level': self.current_level-1,
                    'state_dict': self.agent.model.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                    'success_rate': success_rate
                }, os.path.join(self.checkpoint_path,
                                f'vp_{episode}_{self.current_level}_{round(success_rate, 2)}_{wandb.run.id}.pt'))
                evaluations += 1

                if (evaluations % self.full_evaluation_period) == 0:
                    _, success_rate, path_difference = self.evaluate(levels=range(self.current_level))
                    if log:
                        wandb.log({'full_eval_success_rate': success_rate,
                                   'full_path_difference': path_difference}, step=episode)

            if ((episode_level+1) % self.change_level_every) == 0:
                episode_level = 0
                self.current_level += 1
                if self.current_level > self.end_level:
                    self.current_level = self.start_level
                self.envs = self.load_envs()
                if log:
                    wandb.log({'level': self.current_level}, step=episode)

            if episode == self.max_episodes:
                break

            episode += 1
            episode_level += 1

    def evaluate_model(self, start_level, end_level, log=True, log_animation=False, log_animation_every=10):
        if log:
            wandb.init(project=self.config['project_name'], config=self.config)
        i = 0
        for level in range(start_level, end_level):
            i, completed, path_difference = self.evaluate(levels=[level],
                                                          log_animation=log_animation,
                                                          log_animation_every=log_animation_every,
                                                          counter_start=i)
            metrics = {'success': completed, 'path_difference': path_difference, 'val_level': level}
            if log:
                wandb.log(metrics)
            else:
                print(metrics)


class VPTrainer:
    def __init__(self, config):
        self.config = config
        agent_conf = config['agent']
        self.env_conf = config['environment']

        agent_conf['window'] = self.env_conf['window_size']

        self.evaluate_every_epoch = config['eval_period_epoch']

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.agent = VPAgent(**agent_conf)

        self.checkpoint_name = config['load_checkpoint']
        self.checkpoint_path = config['checkpoint_path']
        self.epoch = 0
        if self.checkpoint_name is not None:
            checkpoint = torch.load(os.path.join(self.checkpoint_path, self.checkpoint_name))
            self.agent.model.load_state_dict(checkpoint['state_dict'])
            self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']

        self.min_level = config['min_level']
        self.max_level = config['max_level']
        self.map_names = config['maps']
        self.sample_every = config['sample_every']
        self.sample_size = config['sample_size']

        if self.map_names is None:
            self.map_names = filter(lambda x: x.split('.')[-1] == 'map', os.listdir(self.env_conf['path']))
            self.map_names = [x.split('.')[0] for x in self.map_names]

        self.max_epoch = config['max_epoch']
        self.success_rate_next_level = config['success_rate_next_level']
        self.current_level = config['min_level']

        self.fill_memory(self.sample_size)

    def load_envs(self, n_maps=None, eval=False, levels=None):
        # 30 maps 30x920 tasks
        # division 6 2 2
        indices = list(range(len(self.map_names)))
        if n_maps is not None:
            indices = random.sample(indices, n_maps)
        if levels is None:
            levels = range(self.min_level, self.max_level)

        envs = list()
        for level in levels:
            if eval:
                start_task = level * 10 + 6
                end_task = level * 10 + 8
            else:
                start_task = level * 10
                end_task = level * 10 + 6

            for i in indices:
                for task in range(start_task, end_task):
                    self.env_conf['task'] = task
                    self.env_conf['map_name'] = self.map_names[i]
                    envs.append(GridWorld(**self.env_conf))
        return envs

    def evaluate(self, levels=None, log_animation=False, log_animation_every=10, counter_start=0):
        envs = self.load_envs(eval=True, levels=levels)
        completed = np.zeros(len(envs))
        path_difference = np.zeros(len(envs))
        i = counter_start
        for j, env in enumerate(envs):
            is_success, duration, v_loss, p_loss, task_length = self.run_episode(env, i,
                                                                                 log_metrics=False,
                                                                                 log_animation=(log_animation and ((i % log_animation_every) == 0)))
            completed[j] = int(is_success)
            if task_length != 0:
                path_difference[j] = (env.path_length - task_length) / task_length

            i += 1

        return i, completed.mean(), path_difference[completed == 1].mean()

    def sample(self, env, sample_size):
        env.reset()
        valid_coords = np.argwhere(np.array(env.map.cells) == 0)
        indices = np.arange(valid_coords.shape[0])
        np.random.shuffle(indices)
        sampled_indices = indices[:sample_size]

        for position in valid_coords[sampled_indices]:
            env.position = position
            obs, reward, is_done = env.observe()
            action_probs = env.get_action_probs()
            action_probs = torch.from_numpy(action_probs[None]).float().to(device=self.device)
            value = env.get_value()
            value = torch.from_numpy(value[None]).float().to(device=self.device)
            window, vector = obs
            window = torch.from_numpy(window[None])
            window = window.float().to(device=self.device)
            vector = torch.from_numpy(vector[None]).float().to(device=self.device)
            # Store state in memory
            self.agent.memory.push(window, vector, action_probs, value)

    def fill_memory(self, samples_per_env=None):
        if samples_per_env is None:
            samples_per_env = self.agent.capacity // (30*(self.max_level+1)*6)
        print('\nFilling the memory...')
        print(f'Filled {len(self.agent.memory)} of {self.agent.memory.capacity}')
        for level in tqdm(range(self.min_level, self.max_level)):
            envs = self.load_envs(levels=[level])
            for env in envs:
                self.sample(env, samples_per_env)

    def train(self, log_metrics=True, log_animation=True, log_animation_every=10):
        if log_metrics:
            run = wandb.init(project=self.config['project_name'], config=self.config)
            wandb.log({'level': self.current_level}, step=0)
        while (self.epoch != self.max_epoch) and (self.current_level <= self.max_level):
            value_loss, policy_loss = self.agent.optimize_model()
            if log_metrics:
                wandb.log({'value_loss': value_loss, 'policy_loss': policy_loss}, step=self.epoch)
            self.epoch += 1

            if ((self.epoch % self.evaluate_every_epoch) == 0) or (self.epoch == self.max_epoch):
                _, success_rate, path_difference = self.evaluate(levels=[self.current_level])
                if log_metrics:
                    wandb.log({'eval_success_rate': success_rate,
                               'path_difference': path_difference}, step=self.epoch)
                if success_rate >= self.success_rate_next_level:
                    self.current_level += 1
                    if log_metrics:
                        wandb.log({'level': self.current_level}, step=self.epoch)

                torch.save({
                    'epoch': self.epoch,
                    'level': self.current_level,
                    'state_dict': self.agent.model.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                    'success_rate': success_rate
                }, os.path.join(self.checkpoint_path,
                                f'az_{self.epoch}_{self.current_level}_{round(success_rate, 2)}_{wandb.run.id}.pt'))

            if (self.epoch % self.sample_every) == 0:
                self.fill_memory(self.sample_size//4)

    def run_episode(self, env, i_episode, log_metrics=False, log_animation=False):
        env.reset()

        total_val_loss = 0
        total_pol_loss = 0
        t = 0
        for t in count():
            obs, reward, is_done = env.observe()
            action_probs = env.get_action_probs()
            action_probs = torch.from_numpy(action_probs[None]).float().to(device=self.device)
            value = env.get_value()
            value = torch.from_numpy(value[None]).float().to(device=self.device)
            window, vector = obs
            window = torch.from_numpy(window[None])
            window = window.float().to(device=self.device)
            vector = torch.from_numpy(vector[None]).float().to(device=self.device)
            # Select and perform an action
            if not is_done:
                action = self.agent.make_action((window, vector), evaluation=True)
                env.act(action)

            if log_animation:
                world, rmap = env.render()
                plt.imsave(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{t}.png', world)

            if is_done:
                break

        if t != 0:
            av_val_loss = total_val_loss / t
            av_pol_loss = total_pol_loss / t
        else:
            av_val_loss = total_val_loss
            av_pol_loss = total_pol_loss

        if log_metrics:
            wandb.log({'duration': t, 'value_loss': av_val_loss, 'policy_loss': av_pol_loss}, step=i_episode)

        if log_animation:
            with imageio.get_writer(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', mode='I', fps=3) as writer:
                for i in range(t):
                    image = imageio.imread(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{i}.png')
                    writer.append_data(image)
            wandb.log({f'animation': wandb.Video(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', fps=3,
                                                 format='gif')}, step=i_episode)

        return env.is_success, t, av_val_loss, av_pol_loss, env.task_length


if __name__ == '__main__':
    import yaml

    with open('../../configs/dqn/vp_curriculum_default.yaml', 'r') as file:
        config = yaml.load(file, yaml.Loader)

    runner = VPAgentCurriculum(config)
    runner.run_curriculum(log_video_every=5000)
    # runner.evaluate_model(0, 20, log_animation=True, log_animation_every=10)

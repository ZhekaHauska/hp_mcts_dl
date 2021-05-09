from mcts_dl.algorithms.mcts import MCTS
from mcts_dl.algorithms.value_policy_network import ValuePolicyNetwork
from mcts_dl.algorithms.model_network import ModelNetworkBorder
from mcts_dl.environment.gridworld_pomdp import GridWorld

import numpy
import torch
import os
import matplotlib.pyplot as plt
import wandb
import imageio
import random
from itertools import count


class Agent:
    def __init__(self, config):
        self.config = config
        self.device = config['device']
        vp_config = config['vp_net']
        om_config = config['om_net']
        self.value_policy_net = ValuePolicyNetwork(**vp_config)
        self.observation_model_net = ModelNetworkBorder(**om_config)

        self.vp_checkpoint_name = config['vp_checkpoint']
        self.vp_checkpoint_path = config['vp_checkpoint_path']
        self.om_checkpoint_name = config['om_checkpoint']
        self.om_checkpoint_path = config['om_checkpoint_path']

        checkpoint = torch.load(os.path.join(self.vp_checkpoint_path, self.vp_checkpoint_name))
        self.value_policy_net.load_state_dict(checkpoint['state_dict'])
        self.value_policy_net.to(device=self.device).eval()

        checkpoint = torch.load(os.path.join(self.om_checkpoint_path, self.om_checkpoint_name))
        self.observation_model_net.load_state_dict(checkpoint['state_dict'])
        self.observation_model_net.to(device=self.device).eval()

        self.mcts = MCTS(config['mcts'])

    def make_action(self, observation):
        root, mcts_info = self.mcts.run(self.value_policy_net,
                                        self.observation_model_net,
                                        observation,
                                        False)

        visit_counts = numpy.array(
            [child.visit_count for child in root.children.values()], dtype="int32"
        )
        actions = [action for action in root.children.keys()]

        action = actions[numpy.argmax(visit_counts)]

        return action


class AgentRunner:
    def __init__(self, config):
        self.config = config
        self.project_name = config['project_name']
        self.entity = config['entity']
        self.agent_config = config['agent']
        self.environment_config = config['environment']
        self.agent = Agent(self.agent_config)
        self.min_level = config['min_level']
        self.max_level = config['max_level']
        self.map_names = config['maps']

    def run_episode(self, env, i_episode, log_metrics=True, log_animation=False):
        env.reset()

        for t in count():
            obs, reward, is_done = env.observe()

            if log_animation:
                world, rmap = env.render()
                plt.imsave(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{t}.png', world)

            if is_done:
                break

            window, vector = obs
            window = torch.from_numpy(window[None])
            window = window.float().to(device=self.agent.device)
            vector = torch.from_numpy(vector[None]).float().to(device=self.agent.device)
            # Select and perform an action
            action = self.agent.make_action((window, vector))
            env.act(action)

        if env.task_length != 0:
            path_difference = (env.path_length - env.task_length) / env.task_length
        else:
            path_difference = 0

        if log_metrics:
            wandb.log({'duration': t, 'success': env.is_success, 'path_diff': path_difference,
                       'task_length': env.task_length}, step=i_episode)

        if log_animation:
            with imageio.get_writer(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', mode='I', fps=3) as writer:
                for i in range(t):
                    image = imageio.imread(f'/tmp/{wandb.run.id}_episode_{i_episode}_step_{i}.png')
                    writer.append_data(image)
            wandb.log({f'animation': wandb.Video(f'/tmp/{wandb.run.id}_episode_{i_episode}.gif', fps=3,
                                                 format='gif')}, step=i_episode)

        return env.is_success, t, path_difference, env.task_length

    def run(self, log_metrics=True, log_animation=True, log_animation_every=100, mode='test'):
        episode = 0
        for level in range(self.min_level, self.max_level):
            envs = self.load_envs([level], mode)
            completed = numpy.zeros(len(envs))
            path_difference = numpy.zeros(len(envs))
            for i, env in enumerate(envs):
                is_success, duration, path_diff, task_length = self.run_episode(env,
                                                                                log_metrics=log_metrics,
                                                                                log_animation=((episode % log_animation_every) == 0) and log_animation,
                                                                                i_episode=episode)
                completed[i] = int(is_success)
                path_difference[i] = path_diff
                episode += 1

            if log_metrics:
                run = wandb.init(project=self.project_name, entity=self.entity, config=self.config)
                wandb.log({'level': level,
                           'success_rate': completed.mean(),
                           'av_path_diff': path_difference[completed == 1].mean()}, step=episode)

    def load_envs(self, levels, n_maps=None, mode='test'):
        indices = list(range(len(self.map_names)))
        if n_maps is not None:
            indices = random.sample(indices, n_maps)

        envs = list()
        for level in levels:
            if mode == 'test':
                start_task = level * 10 + 8
                end_task = level * 10 + 10
            elif mode == 'eval':
                start_task = level * 10 + 6
                end_task = level * 10 + 8
            elif mode == 'train':
                start_task = level * 10
                end_task = level * 10 + 6
            else:
                raise ValueError(f'There is no such mode: {mode}! Possible modes: "test", "eval" and "train"')

            for i in indices:
                for task in range(start_task, end_task):
                    self.environment_config['task'] = task
                    self.environment_config['map_name'] = self.map_names[i]
                    envs.append(GridWorld(**self.environment_config))
        return envs

import os.path
import numpy as np
from math import atan, pi, tan, sqrt

from mcts_dl.utils.utils import Map, ReadMapFromMovingAIFile, ReadTasksFromMovingAIFile


class GridWorld:
    map: Map

    def __init__(self, map_name, task, window_size, max_steps_mult=3, move_reward=0, goal_reward=1,
                 distance_reward_weight_forward=1, distance_reward_weight_backward=-1,
                 rest_distance_reward_weight=0, collision_reward=0, path='.',
                 max_steps=None,
                 disable_repetitions=False):
        self.collision_reward = collision_reward
        self.move_reward = move_reward
        self.map_name = map_name
        self.path = path

        tasks = ReadTasksFromMovingAIFile(f'{path}/{map_name}.map.scen')
        j_start, i_start, j_goal, i_goal, length = tasks[task]
        self.map = ReadMapFromMovingAIFile(os.path.join(self.path, self.map_name + '.map'))

        self.window_size = window_size
        self.task_length = length
        self.position = np.array([i_start, j_start])
        self.previous_position = None
        self.start_position = np.array([i_start, j_start])
        self.goal_position = np.array([i_goal, j_goal])
        self.actions = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                                 [0, 1], [0, -1], [1, 0], [-1, 0]])
        self.vec = self.goal_position - self.start_position
        self.max_length = self.map.width * (2 ** 0.5)
        self.signal = np.linalg.norm(self.vec) / self.max_length
        self.reward = 0
        self.value = 0
        self.action_probs = np.ones(self.actions.shape[0]) / self.actions.shape[0]
        self.goal_reward = goal_reward
        self.distance_reward_weight_forward = distance_reward_weight_forward
        self.distance_reward_weight_backward = distance_reward_weight_backward
        self.rest_distance_reward_weight = rest_distance_reward_weight
        if max_steps is None:
            self.max_steps = length * max_steps_mult
        else:
            self.max_steps = max_steps
        self.step = 0

        self.reward_map = np.load(
            os.path.join(self.path, self.map_name, f'{i_goal}_{j_goal}.rmap.npy')
        )
        self.reward_map /= self.reward_map.max()
        self.reward_map = 1 - self.reward_map

        self.done = False
        self.is_success = False
        if np.all(self.position == self.goal_position):
            self.done = True
            self.is_success = True
        self.path_length = 0
        self.disable_repetitions = disable_repetitions

    def act(self, action):
        new_position = self.position + self.actions[action]
        self.reward = 0
        if tuple(new_position) in self.map.GetNeighbors(*self.position):
            if self.disable_repetitions and np.all(new_position == self.previous_position):
                self.done = True
                self.reward = self.collision_reward
            else:
                if np.all(new_position == self.goal_position):
                    self.done = True
                    self.is_success = True
                    self.reward = self.goal_reward
                else:
                    distance_reward = (self.reward_map[self.position[0], self.position[1]] -
                                       self.reward_map[new_position[0], new_position[1]]
                                       )
                    if distance_reward <= 0:
                        distance_reward *= self.distance_reward_weight_backward
                    else:
                        distance_reward *= self.distance_reward_weight_forward

                    self.reward = distance_reward + self.move_reward

                self.previous_position = self.position
                self.position = new_position
                self.vec = self.goal_position - self.position
                self.signal = np.linalg.norm(self.vec) / self.max_length
                self.path_length += np.linalg.norm(self.actions[action])
        else:
            self.done = True
            self.reward = self.collision_reward

        rest_distance_reward = self.reward_map[self.position[0], self.position[1]]
        rest_distance_reward *= self.rest_distance_reward_weight
        self.reward += rest_distance_reward

        self.step += 1
        if self.step >= self.max_steps:
            self.done = True

    def get_action_probs(self):
        self.action_probs = np.zeros(self.actions.shape[0])
        positions = self.position + self.actions
        neighbours = self.map.GetNeighbors(*self.position)
        values = np.zeros(self.actions.shape[0]) - 1
        for i, pos in enumerate(positions):
            if tuple(pos) in neighbours:
                values[i] = self.reward_map[pos[0], pos[1]]
        self.action_probs[np.argmax(values)] = 1
        self.action_probs /= self.action_probs.sum()
        return self.action_probs

    def get_value(self):
        self.value = self.reward_map[self.position[0], self.position[1]]
        return self.value

    def reset(self):
        self.step = 0
        self.position = self.start_position
        self.previous_position = None
        self.done = False
        self.is_success = False
        self.reward = 0
        self.value = 0
        self.vec = self.goal_position - self.start_position
        self.action_probs = np.ones(self.actions.shape[0]) / self.actions.shape[0]
        self.signal = np.linalg.norm(self.vec) / self.max_length
        self.path_length = 0

    def observe(self):
        window = self.get_window(np.array(self.map.cells), self.position, self.window_size)
        vec = np.zeros(2)
        vec[0] = atan(self.vec[0] / (self.vec[1] + 1e-12))
        if self.vec[1] < 0 < self.vec[0]:
            vec[0] += pi
        elif (self.vec[1] < 0) and (self.vec[0] <= 0):
            vec[0] -= pi
        vec[0] /= pi
        vec[1] = self.signal
        return (window, vec), self.reward, self.done

    def get_window(self, map, position, window_size: int):
        world = map[None]
        row = position[0]
        col = position[1]

        top = row - window_size // 2
        bottom = row + window_size // 2 + 1

        if top < 0:
            rows = np.concatenate((np.ones_like(world[:, top:, :]), world[:, 0: bottom, :]), axis=1)
        elif bottom >= world.shape[1]:
            rows = np.concatenate((world[:, top:, :], np.ones_like(world[:, 0: bottom - world.shape[1], :])), axis=1)
        else:
            rows = world[:, top: bottom, :]

        left = col - window_size // 2
        right = col + window_size // 2 + 1
        if left < 0:
            cols = np.concatenate((np.ones_like(rows[:, :, left:]), rows[:, :, 0: right]), axis=2)
        elif right >= world.shape[2]:
            cols = np.concatenate((rows[:, :, left:], np.ones_like(rows[:, :, 0: right - world.shape[2]])), axis=2)
        else:
            cols = rows[:, :, left: right]
        return cols

    def render(self):
        world = np.array(self.map.cells)
        row = self.position[0]
        col = self.position[1]
        world[row, col] = 2
        world[self.goal_position[0], self.goal_position[1]] = 3
        window = self.get_window(world, self.position, self.window_size)[0]
        rmap = self.get_window(self.reward_map, self.position, self.window_size)[0]
        return window, rmap

    def in_bounds(self, position):
        return (0 <= position[0] < self.reward_map.shape[0]) and (0 <= position[1] < self.reward_map.shape[1])


def calculate_next_vector(vector, displacement, max_length):
    """
    Calculates new observation vector for GridWorld.
    :param vector: (direction, distance), where direction -- alpha/pi, distance -- |v|/max_length
    :param displacement: (delta_row, delta_col) -- agent displacement
    :param max_length: sqrt(2)*map_width
    :return: new observation vector (direction, distance)
    """
    # recover vector to goal
    alpha = vector[0] * pi
    d = tan(alpha)
    col = max_length * vector[1] / sqrt(d ** 2 + 1)
    row = d * col
    if abs(alpha) > pi / 2:
        col = -col

    new_row = row - displacement[0]
    new_col = col - displacement[1]
    # get vector back again
    new_vec = np.zeros(2)
    new_vec[0] = atan(new_row / (new_col + 1e-12))
    if new_col < 0 < new_row:
        new_vec[0] += pi
    elif (new_col < 0) and (new_row <= 0):
        new_vec[0] -= pi
    new_vec[0] /= pi
    new_vec[1] = sqrt(new_col ** 2 + new_row ** 2) / max_length
    return new_vec


def is_move_possible(obs_window, displacement):
    """
    Check move possibility using observation window
    :param obs_window: (height, width) 2D image
    :param displacement: (delta_row, delta_col) -- agent displacement
    :return: True if move is possible, False -- otherwise
    """
    pos_row = obs_window.shape[0] // 2
    pos_col = obs_window.shape[1] // 2

    new_pos_row = pos_row + displacement[0]
    new_pos_col = pos_col + displacement[1]

    is_traversable = obs_window[new_pos_row, new_pos_col] == 0
    is_shortcut = obs_window[new_pos_row, pos_col] or obs_window[pos_row, new_pos_col]
    return is_traversable and (not is_shortcut)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = '../../dataset/256'
    map_name = 'Berlin_0_256'
    task = 50
    window = 21
    env = GridWorld('Berlin_0_256', task, window, path=path)

    plt.imshow(env.map.cells)
    plt.show()
    obs = env.observe()
    plt.imshow(obs[0][0].reshape((window, window)))
    plt.show()
    for direction in range(8):
        env.reset()
        obs, _, _ = env.observe()
        obs_vec = obs[1]
        for action in [direction] * 10:
            pred_obs_vec = calculate_next_vector(obs_vec, env.actions[action], env.max_length)
            env.act(action)
            obs, reward, done = env.observe()
            window_obs, obs_vec = obs
            value = env.get_value()
            probs = env.get_action_probs()

            print('-'*10)
            print(f'displacement: {env.actions[action]}')
            print(f'value {value}, probs: {probs}')
            print(f'vector {env.vec}')
            print(f'obs vector: {obs_vec}')
            print(f'predicted obs vector: {pred_obs_vec}')
            print(f'prediction error: {obs_vec - pred_obs_vec}')


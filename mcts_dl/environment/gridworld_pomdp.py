import os.path
import numpy as np

from mcts_dl.utils.utils import Map, ReadMapFromMovingAIFile, ReadTasksFromMovingAIFile


class GridWorld:
    map: Map

    def __init__(self, map_name, task, window_size, goal_reward=1, distance_reward_weight=-1, path='.', max_steps=None):
        self.map_name = map_name
        self.path = path

        tasks = ReadTasksFromMovingAIFile(f'{path}/{map_name}.map.scen')
        j_start, i_start, j_goal, i_goal, length = tasks[task]
        self.map = ReadMapFromMovingAIFile(os.path.join(self.path, self.map_name + '.map'))

        self.window_size = window_size
        self.done = False
        self.task_length = length
        self.position = np.array([i_start, j_start])
        self.start_position = np.array([i_start, j_start])
        self.goal_position = np.array([i_goal, j_goal])
        self.actions = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                                 [0, 1], [0, -1], [1, 0], [-1, 0]])
        self.vec = self.goal_position - self.start_position
        self.vec = self.vec / self.task_length
        self.reward = 0
        self.goal_reward = goal_reward
        self.distance_reward_weight = distance_reward_weight
        if max_steps is None:
            self.max_steps = length * 10
        else:
            self.max_steps = max_steps
        self.step = 0

        self.reward_map = np.load(
            os.path.join(self.path, self.map_name, f'{i_goal}_{j_goal}.rmap.npy')
        )

    def act(self, action):
        new_position = self.position + self.actions[action]
        self.reward = 0
        if self.in_bounds(new_position) and (self.reward_map[new_position[0], new_position[1]] != -1):
            self.position = new_position

            self.vec = self.goal_position - self.position
            self.vec = self.vec / self.task_length
            if self.reward_map[self.position[0], self.position[1]] == 0:
                self.done = True
                self.reward = self.goal_reward

        self.reward += self.distance_reward_weight * (self.reward_map[self.position[0], self.position[1]] / self.task_length)

        self.step += 1
        if self.step >= self.max_steps:
            self.done = True

    def reset(self):
        self.step = 0
        self.position = self.start_position
        self.done = False
        self.reward = 0
        self.vec = self.goal_position - self.start_position

    def observe(self):
        window = self.get_window(np.array(self.map.cells), self.position, self.window_size)
        return (window, self.vec), self.reward, self.done

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
        window = self.get_window(world, self.position, int(self.task_length * 2))[0]
        rmap = self.get_window(self.reward_map, self.position, int(self.task_length * 2))[0]
        return window, rmap

    def in_bounds(self, position):
        return (0 <= position[0] < self.reward_map.shape[0]) and (0 <= position[1] < self.reward_map.shape[1])


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    path = '../../dataset/256'
    map_name = 'Berlin_0_256'
    task = 100
    window = 21
    env = GridWorld('Berlin_0_256', task, window, path=path)

    plt.imshow(env.map.cells)
    plt.show()
    obs = env.observe()
    plt.imshow(obs[0][0].reshape((window, window)))
    plt.show()

    for action in [0, 1, 2, 3, 4, 5, 6, 7]:
        env.act(action)
        obs = env.observe()
        plt.imshow(obs[0][0].reshape((window, window)))
        plt.show()

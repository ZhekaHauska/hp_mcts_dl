import os.path
import numpy as np

from mcts_dl.utils.utils import Map, ReadMapFromMovingAIFile


class GridWorld:
    map: Map

    def __init__(self, map_name, i_start, j_start, i_goal, j_goal, window_size, path='.'):
        self.map_name = map_name
        self.path = path

        self.map = ReadMapFromMovingAIFile(os.path.join(self.path, self.map_name + '.map'))

        self.window_size = window_size
        self.done = False
        self.position = np.array(i_start, j_start)
        self.start_position = np.array(i_start, j_start)
        self.goal_position = np.array(i_goal, j_goal)
        self.actions = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                                 [0, 1], [0, -1], [1, 0], [-1, 0]])
        self.vec = self.goal_position - self.start_position
        self.reward = 0

        self.reward_map = np.load(
            os.path.join(self.path, self.map.height, self.map_name, f'{i_goal}_{j_goal}.rmap.npy')
        )

    def act(self, action):
        new_position = self.position + self.actions[action]
        self.reward = self.reward_function(self.position, new_position)

        if new_position in self.map.GetNeighbors(*self.position):
            self.position = new_position
            self.vec = self.goal_position - self.position
            if self.position == self.goal_position:
                self.done = True

    def reset(self):
        self.position = self.start_position
        self.done = False

    def observe(self):
        window = self.get_window()
        return (window, self.vec), self.reward, self.done

    def get_window(self):
        world = np.array(self.map.cells)
        row = self.position[1]
        col = self.position[0]

        top = row - self.window_size // 2
        bottom = row + self.window_size // 2 + 1
        if top < 0:
            rows = np.concatenate((world[:, top:, :], world[:, 0: bottom, :]), axis=1)
        elif bottom >= world.shape[1]:
            rows = np.concatenate((world[:, top:, :], world[:, 0: bottom - world.shape[1], :]), axis=1)
        else:
            rows = world[:, top: bottom, :]

        left = col - self.window_size // 2
        right = col + self.window_size // 2 + 1
        if left < 0:
            cols = np.concatenate((rows[:, :, left:], rows[:, :, 0: right]), axis=2)
        elif right >= world.shape[2]:
            cols = np.concatenate((rows[:, :, left:], rows[:, :, 0: right - world.shape[2]]), axis=2)
        else:
            cols = rows[:, :, left: right]
        return cols

    def reward_function(self, pos, next_pos):
        return self.reward_map[pos] - self.reward_map[next_pos]



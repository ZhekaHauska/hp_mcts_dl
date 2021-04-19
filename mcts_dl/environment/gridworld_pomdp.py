import os.path
import numpy as np

from mcts_dl.utils.utils import Map, ReadMapFromMovingAIFile


class GridWorld:
    map: Map

    def __init__(self, map_name, i_start, j_start, i_goal, j_goal, window_size, collision_reward=-1,
                 move_reward=-0.1, path='.'):
        self.map_name = map_name
        self.path = path

        self.map = ReadMapFromMovingAIFile(os.path.join(self.path, self.map_name + '.map'))

        self.window_size = window_size
        self.done = False
        self.position = np.array([i_start, j_start])
        self.start_position = np.array([i_start, j_start])
        self.goal_position = np.array([i_goal, j_goal])
        self.actions = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1],
                                 [0, 1], [0, -1], [1, 0], [-1, 0]])
        self.vec = self.goal_position - self.start_position
        self.reward = 0
        self.collision_reward = collision_reward
        self.move_reward = move_reward

        self.reward_map = np.load(
            os.path.join(self.path, self.map_name, f'{i_goal}_{j_goal}.rmap.npy')
        )

    def act(self, action):
        new_position = self.position + self.actions[action]
        if self.reward_map[new_position[0], new_position[1]] == -1:
            self.reward = self.collision_reward
        else:
            self.reward = self.reward_map[self.position[0],
                                          self.position[1]] - self.reward_map[new_position[0],
                                                                              new_position[1]]
            self.reward += self.move_reward
            self.position = new_position
            self.vec = self.goal_position - self.position
            if self.reward_map[self.position[0], self.position[1]] == 0:
                self.done = True

    def reset(self):
        self.position = self.start_position
        self.done = False
        self.reward = 0
        self.vec = self.goal_position - self.start_position

    def observe(self):
        window = self.get_window()
        return (window, self.vec), self.reward, self.done

    def get_window(self):
        world = np.array(self.map.cells)[None]
        row = self.position[0]
        col = self.position[1]

        top = row - self.window_size // 2
        bottom = row + self.window_size // 2 + 1

        if top < 0:
            rows = np.concatenate((np.ones_like(world[:, top:, :]), world[:, 0: bottom, :]), axis=1)
        elif bottom >= world.shape[1]:
            rows = np.concatenate((world[:, top:, :], np.ones_like(world[:, 0: bottom - world.shape[1], :])), axis=1)
        else:
            rows = world[:, top: bottom, :]

        left = col - self.window_size // 2
        right = col + self.window_size // 2 + 1
        if left < 0:
            cols = np.concatenate((np.ones_like(rows[:, :, left:]), rows[:, :, 0: right]), axis=2)
        elif right >= world.shape[2]:
            cols = np.concatenate((rows[:, :, left:], np.ones_like(rows[:, :, 0: right - world.shape[2]])), axis=2)
        else:
            cols = rows[:, :, left: right]
        return cols


if __name__ == '__main__':
    from mcts_dl.utils.utils import ReadTasksFromMovingAIFile
    import matplotlib.pyplot as plt

    path = '../../dataset/256'
    map_name = 'Berlin_0_256'
    tasks = ReadTasksFromMovingAIFile(f'{path}/{map_name}.map.scen')
    window = 21
    j_start, i_start, j_goal, i_goal, length = tasks[100]
    env = GridWorld('Berlin_0_256', i_start, j_start, i_goal, j_goal, window,
                    collision_reward=-1, path=path)

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

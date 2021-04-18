import math
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from mcts_dl.algorithms.dijkstra import Dijkstra


def CalculateCost(i1, j1, i2, j2):
    return math.sqrt((i1 - i2) ** 2 + (j1 - j2) ** 2)


class Node:
    def __init__(self, i=-1, j=-1, g=math.inf, h=math.inf, F=None, parent=None):
        self.i = i
        self.j = j
        self.g = g
        if F is None:
            self.F = self.g + h
        else:
            self.F = F
        self.parent = parent

    def __eq__(self, other):
        return (self.i == other.i) and (self.j == other.j)


class Map:
    def __init__(self):
        self.map_name = ''
        self.width = 0
        self.height = 0
        self.cells = []

    # Initialization of map by string.
    def ReadFromString(self, cellStr, width, height, obstacle=None, free=None):
        if free is None:
            free = {'.'}

        if obstacle is None:
            obstacle = {'@'}

        self.width = width
        self.height = height
        self.cells = [[0 for _ in range(width)] for _ in range(height)]
        cellLines = cellStr.split("\n")
        i = 0
        j = 0
        for l in cellLines:
            if len(l) != 0:
                j = 0
                for c in l:
                    if c in free:
                        self.cells[i][j] = 0
                    elif c in obstacle:
                        self.cells[i][j] = 1
                    else:
                        continue
                    j += 1
                # TODO
                if j != width:
                    raise Exception("Size Error. Map width = ", j, ", but must be", width)

                i += 1

        if i != height:
            raise Exception("Size Error. Map height = ", i, ", but must be", height)

    # Initialization of map by list of cells.
    def SetGridCells(self, width, height, gridCells):
        self.width = width
        self.height = height
        self.cells = gridCells

    # Checks cell is on grid.
    def inBounds(self, i, j):
        return (0 <= j < self.width) and (0 <= i < self.height)

    # Checks cell is not obstacle.
    def Traversable(self, i, j):
        return not self.cells[i][j]

    # Creates a list of neighbour cells as (i,j) tuples.
    def GetNeighbors(self, i, j):
        neighbors = []
        delta = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        for d in delta:
            if self.inBounds(i + d[0], j + d[1]) and self.Traversable(i + d[0], j + d[1]):
                neighbors.append((i + d[0], j + d[1]))

        delta = [[1, 1], [1, -1], [-1, -1], [-1, 1]]

        for d in delta:
            if (self.inBounds(i + d[0], j + d[1]) and self.Traversable(i + d[0], j + d[1]) and
                    (self.Traversable(i + d[0], j + 0) and self.Traversable(i + 0, j + d[1]))):
                neighbors.append((i + d[0], j + d[1]))

        return neighbors


def MakePath(goal):
    length = goal.g
    current = goal
    path = []
    while current.parent:
        path.append(current)
        current = current.parent
    path.append(current)
    return path[::-1], length


def ReadMapFromMovingAIFile(path):
    with open(path, 'r') as file:
        file.readline()
        height = int(file.readline().split()[1])
        width = int(file.readline().split()[1])
        file.readline()
        map_str = file.read()
    taskMap = Map()
    taskMap.ReadFromString(map_str, width, height)

    return taskMap


def ReadTasksFromMovingAIFile(path):
    tasks = []

    with open(path, 'r') as file:
        file.readline()
        lines = file.readlines()

    for line in lines:
        items = line.split()[-5:]
        tasks.append([int(x) for x in items[:-1]] + [float(items[-1])])

    return tasks


def ComputeRmap(map_name: str, path='.', silent=False, **args):
    tasks = ReadTasksFromMovingAIFile(f'{path}/{map_name}.map.scen')
    taskMap = ReadMapFromMovingAIFile(f'{path}/{map_name}.map')
    goals = set([(task[3], task[2]) for task in tasks])
    Path(f'{path}/{map_name}/').mkdir(parents=True, exist_ok=True)
    if not silent:
        print(f'map: {map_name}')
    for iGoal, jGoal in tqdm(goals, disable=silent):
        closed = Dijkstra(taskMap, iGoal, jGoal, **args)
        rmap = np.zeros((taskMap.width, taskMap.height)) - 1
        for coords, node in closed.elements.items():
            rmap[coords[0], coords[1]] = node.g
        np.save(f'{path}/{map_name}/{iGoal}_{jGoal}.rmap', rmap)


def test(*args, **kwargs):
    print(args, kwargs)
    time.sleep(10)


def ComputeRmapsParallel(map_list: list[str], n_jobs=-1, path='.', **args):
    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    n_batches = round(len(map_list) / n_jobs)
    for batch in tqdm(range(n_batches)):
        processes = list()
        for map_name in map_list[batch*n_jobs: (batch+1)*n_jobs]:
            p = multiprocessing.Process(target=ComputeRmap, args=(map_name, path, True), kwargs=args)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()


if __name__ == '__main__':
    import time
    start_time = time.time()

    maps = os.listdir('../../dataset/256/')
    maps = [x.split('.')[0] for x in maps if x.endswith('.map')]

    ComputeRmapsParallel(maps, path='../../dataset/256')
    print("--- %s seconds ---" % (time.time() - start_time))

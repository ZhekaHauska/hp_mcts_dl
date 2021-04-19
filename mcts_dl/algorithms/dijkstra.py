import heapq as hq
import os
import numpy as np
from pathlib import Path
from tqdm import tqdm
import multiprocessing

from mcts_dl.utils.utils import Node, Map, CalculateCost, ReadTasksFromMovingAIFile, ReadMapFromMovingAIFile


class Open:
    def __init__(self):
        self.heap = list()
        self.mapping = dict()
        self.removed_flag = 'removed'
        self.counter = 0

    def __iter__(self):
        return iter([item[-1] for item in self.mapping.values()])

    def __len__(self):
        return len(self.mapping)

    def isEmpty(self):
        if len(self.mapping) != 0:
            return False
        return True

    def GetBestNode(self):
        while self.heap:
            F, g, c, node = hq.heappop(self.heap)
            if node is not self.removed_flag:
                del self.mapping[(node.i, node.j)]
                return node
        else:
            return None

    def AddNode(self, item: Node):
        coords = (item.i, item.j)
        if coords in self.mapping:
            if item.g < self.mapping[coords][-1].g:
                self._RemoveNode(coords)
            else:
                return

        self.counter += 1
        entry = [item.F, item.g, -self.counter, item]
        self.mapping[coords] = entry
        hq.heappush(self.heap, entry)

    def _RemoveNode(self, coords):
        entry = self.mapping.pop(coords)
        entry[-1] = self.removed_flag


class Closed:
    def __init__(self):
        self.elements = dict()

    def __iter__(self):
        return iter(self.elements.values())

    def __len__(self):
        return len(self.elements.values())

    # AddNode is the method that inserts the node to CLOSED
    def AddNode(self, item: Node, *args):
        self.elements[(item.i, item.j)] = item

    # WasExpanded is the method that checks if a node has been expanded
    def WasExpanded(self, item: Node, *args):
        return (item.i, item.j) in self.elements


def Dijkstra(gridMap: Map, iStart: int, jStart: int, openType=Open, closedType=Closed):
    """
    Calculates distance from iStart, jStart to all cells on gridMap
    :param gridMap:
    :param iStart:
    :param jStart:
    :param openType:
    :param closedType:
    :return:
    """
    OPEN = openType()
    CLOSED = closedType()

    node = Node(iStart, jStart, g=0, h=0)
    OPEN.AddNode(node)
    while not OPEN.isEmpty():
        node = OPEN.GetBestNode()

        neighbors = gridMap.GetNeighbors(node.i, node.j)
        for neighbor in neighbors:
            if not CLOSED.WasExpanded(Node(*neighbor)):
                g = node.g + CalculateCost(node.i, node.j,
                                           *neighbor)
                node_n = Node(*neighbor, g=g, h=0, parent=node)
                OPEN.AddNode(node_n)

        CLOSED.AddNode(node)

    return CLOSED


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
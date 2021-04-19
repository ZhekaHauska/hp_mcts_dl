import math


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

import heapq as hq
from mcts_dl.utils.utils import Node, Map, CalculateCost


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

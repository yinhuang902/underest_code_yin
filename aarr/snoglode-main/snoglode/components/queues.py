import heapq
from collections.abc import Iterable
from enum import Enum


class QueueStrategy(Enum):
    lifo = 'lifo'
    fifo = 'fifo'
    bound = 'bound'



class NodeQueue(Iterable):
    def __init__(self) -> None:
        self._q = []

    def __iter__(self):
        for i in self._q:
            yield i

    def __len__(self):
        return len(self._q)
    
    def push(self, node):
        raise NotImplementedError('should be implemented by derived classes')
    
    def pop(self):
        raise NotImplementedError('should be implemented by derived classes')


class LIFONodeQueue(NodeQueue):
    def __init__(self) -> None:
        super().__init__()
        self._ndx = 0

    def push(self, node):
        heapq.heappush(self._q, (self._ndx, node))
        self._ndx -= 1

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node


class FIFONodeQueue(NodeQueue):
    def __init__(self) -> None:
        super().__init__()
        self._ndx = 0

    def push(self, node):
        heapq.heappush(self._q, (self._ndx, node))
        self._ndx += 1

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node


class WorstBoundNodeQueue(NodeQueue):
    def push(self, node):
        heapq.heappush(self._q, (node.lb_problem.objective, node))

    def pop(self):
        _, node = heapq.heappop(self._q)
        return node

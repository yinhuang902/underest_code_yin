from snoglode.components.queues import LIFONodeQueue, FIFONodeQueue, WorstBoundNodeQueue
from unittest import TestCase


class MockLBProblem:
    def __init__(self, obj) -> None:
        self.objective = obj


class MockNode:
    def __init__(self, obj=None) -> None:
        self.lb_problem = MockLBProblem(obj)


class TestNodeQueues(TestCase):
    def test_lifo(self):
        q = LIFONodeQueue()
        a = [MockNode() for i in range(5)]
        for i in a:
            q.push(i)
        self.assertIs(a[-1], q.pop())
        b = MockNode()
        q.push(b)
        self.assertIs(b, q.pop())
        self.assertIs(a[-2], q.pop())
        self.assertIs(a[-3], q.pop())
        self.assertIs(a[-4], q.pop())
        self.assertIs(a[0], q.pop())

    def test_fifo(self):
        q = FIFONodeQueue()
        a = [MockNode() for i in range(5)]
        for i in a:
            q.push(i)
        self.assertIs(a[0], q.pop())
        b = MockNode()
        q.push(b)
        self.assertIs(a[1], q.pop())
        self.assertIs(a[2], q.pop())
        self.assertIs(a[3], q.pop())
        self.assertIs(a[4], q.pop())
        self.assertIs(b, q.pop())

    def test_worst_bound(self):
        q = WorstBoundNodeQueue()
        n1 = MockNode(7)
        n2 = MockNode(3)
        n3 = MockNode(15)
        n4 = MockNode(5)
        n5 = MockNode(8)
        n6 = MockNode(2)
        q.push(n1)
        q.push(n2)
        q.push(n3)
        q.push(n4)
        q.push(n5)
        self.assertIs(n2, q.pop())
        q.push(n6)
        self.assertIs(n6, q.pop())
        self.assertIs(n4, q.pop())
        self.assertIs(n1, q.pop())
        self.assertIs(n5, q.pop())
        self.assertIs(n3, q.pop())

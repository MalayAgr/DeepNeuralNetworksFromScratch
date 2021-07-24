from .core import ComputationGraph

graph = ComputationGraph()


def reset_graph():
    global graph
    graph = ComputationGraph()

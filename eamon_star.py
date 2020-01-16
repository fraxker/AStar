import networkx as nx
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from numba import cuda, jit

# Cost for doing different actions, used to calculate G and H scores
straight_cost = 1  # Distance of going straight
first_diag_cost = (
    1.41421356  # Distance of going diagonal on a square, estimate of sqrt(2)
)
sec_diag_cost = 1.7320508  # Distance of going diagonal on a cube, estimate of sqrt(3)
size = (10, 100, 100)  # (z, y, x)

def heuristic(node, end_node):
    dx = abs(node[0] - end_node[0])
    dy = abs(node[1] - end_node[1])
    dz = abs(node[2] - end_node[2])
    dmin = min(dx, dy, dz)
    dmax = max(dx, dy, dz)
    dmid = dx + dy + dz - dmin - dmax
    return (
        (sec_diag_cost - first_diag_cost) * dmin
        + (first_diag_cost - straight_cost) * dmid
        + dmax * first_diag_cost
    )

def add_edges(graph):
    first_diag = [
        (-1, -1, 0),
        (-1, 1, 0),
        (1, -1, 0),
        (1, 1, 0),
        (0, -1, -1),
        (0, 1, -1),
        (-1, 0, -1),
        (1, 0, -1),
        (0, -1, 1),
        (0, 1, 1),
        (-1, 0, 1),
        (1, 0, 1),
    ]
    second_diag = [
        (-1, -1, -1),
        (-1, 1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, -1, 1),
        (-1, 1, 1),
        (1, -1, 1),
        (1, 1, 1),
    ]
    for node in list(graph):
        for new_position in first_diag:  # 2D Diagonal edges
            node_position = (
                node[0] + new_position[0],
                node[1] + new_position[1],
                node[2] + new_position[2],
            )

            # Make sure within range
            if (
                node_position[0] > (size[2] - 1)
                or node_position[0] < 0
                or node_position[1] > (size[1] - 1)
                or node_position[1] < 0
                or node_position[2] < 0
                or node_position[2] > (size[0] - 1)
            ):
                continue

            graph.add_edge(
                node,
                (node_position[0], node_position[1], node_position[2],),
                weight=first_diag_cost,
            )
        for new_position in second_diag:  # 2D Diagonal edges
            node_position = (
                node[0] + new_position[0],
                node[1] + new_position[1],
                node[2] + new_position[2],
            )

            if (
                node_position[0] > (size[2] - 1)
                or node_position[0] < 0
                or node_position[1] > (size[1] - 1)
                or node_position[1] < 0
                or node_position[2] < 0
                or node_position[2] > (size[0] - 1)
            ):
                continue

            graph.add_edge(
                node,
                (node_position[0], node_position[1], node_position[2],),
                weight=sec_diag_cost,
            )

def remove_obstacles(obstacles, graph):
    for o in obstacles:
        graph.remove_node(o)

def main():
    grapht0 = time.process_time()
    G = nx.grid_graph(list(size))
    obstacles = [
        (4, 0, 0),
        (4, 1, 0),
        (4, 2, 0),
        (4, 3, 0),
        (4, 6, 0),
        (4, 7, 0),
        (4, 8, 0),
        (4, 9, 0),
        (4, 0, 1),
        (4, 1, 1),
        (4, 2, 1),
        (4, 3, 1),
        (4, 6, 1),
        (4, 7, 1),
        (4, 8, 1),
        (4, 9, 1),
        (4, 0, 2),
        (4, 1, 2),
        (4, 2, 2),
        (4, 3, 2),
        (4, 6, 2),
        (4, 7, 2),
        (4, 8, 2),
        (4, 9, 2),
    ]
    start = (0, 0, 0)
    end = (9, 0, 2)
    add_edges(G)
    remove_obstacles(obstacles, G)
    nx.freeze(G)
    grapht1 = time.process_time()
    print("Graph:", (grapht1 - grapht0))

    patht0 = time.process_time()
    path = nx.astar_path(G, start, end, heuristic=heuristic)
    patht1 = time.process_time()
    print("AStar:", (patht1 - patht0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for o in obstacles:
        ax.scatter(o[0], o[1], o[2], marker="o", c="blue")

    for point in path:
        ax.scatter(
            point[0], point[1], point[2], marker="o", c="magenta",
        )

    ax.scatter(start[0], start[1], start[2], marker="o", c="red")
    ax.scatter(end[0], end[1], end[2], marker="o", c="green")

    ax.set_xlabel("X Label")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    plt.show()


if __name__ == "__main__":
    main()

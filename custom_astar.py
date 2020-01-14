from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import heapq
import math
import numpy as np


class Node:
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None, delta=None):
        self.parent = parent
        self.position = position
        self.delta = delta

        self.g = 0
        self.h = 0
        self.f

    @property
    def f(self):
        return self.g + self.h

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.position == other.position

    def __hash__(self):
        return hash(self.position)


def astar(maze, start, end):
    """Returns a list of tuples as a path from the given start to the given end in the given maze"""

    #Cost for doing different actions, used to calculate G and H scores
    straight_cost = 1 #Distance of going straight
    first_diag_cost = 1.4142 #Distance of going diagonal on a square, estimate of sqrt(2)
    sec_diag_cost = 1.732 #Distance of going diagonal on a cube, estimate of sqrt(3)

    def heuristic(node):
        dx = abs(node.position[0] - end_node.position[0])
        dy = abs(node.position[1] - end_node.position[1])
        dz = abs(node.position[2] - end_node.position[2])
        dmin = min(dx, dy, dz)
        dmax = max(dx, dy, dz)
        dmid = dx + dy + dz - dmin - dmax
        return (sec_diag_cost - first_diag_cost) * dmin + (first_diag_cost - straight_cost) * dmid + dmax * first_diag_cost 

    def delta(position):
        sum = abs(position[0]) + abs(position[1]) + abs(position[3])
        if sum == 3:
            return sec_diag_cost
        else if sum == 2:
            return first_diag_cost
        else:
            return straight_cost


    # Create start and end node
    start_node = Node(None, start)
    end_node = Node(None, end)

    # Initialize both open and closed list
    open_list = []
    closed_list = set()
    heapq.heapify(open_list)

    # Add the start node
    heapq.heappush(open_list, start_node)

    # Loop until you find the end
    while len(open_list) > 0:

        # Get the current node
        current_node = heapq.heappop(open_list)
        closed_list.add(current_node)

        # Found the goal
        if current_node == end_node:
            path = []
            current = current_node
            while current is not None:
                path.append(current.position)
                current = current.parent
            return path[::-1]  # Return reversed path

        # Generate children
        children = []
        for new_position in [
            (0, -1, 0),
            (0, 1, 0),
            (-1, 0, 0),
            (1, 0, 0),
            (-1, -1, 0),
            (-1, 1, 0),
            (1, -1, 0),
            (1, 1, 0),
            (0, -1, -1),
            (0, 1, -1),
            (-1, 0, -1),
            (1, 0, -1),
            (-1, -1, -1),
            (-1, 1, -1),
            (1, -1, -1),
            (1, 1, -1),
            (0, 0 , -1),
            (0, -1, 1),
            (0, 1, 1),
            (-1, 0, 1),
            (1, 0, 1),
            (-1, -1, 1),
            (-1, 1, 1),
            (1, -1, 1),
            (1, 1, 1),
            (0 , 0, 1),
        ]:  # Adjacent squares

            # Get node position
            node_position = (
                current_node.position[0] + new_position[0],
                current_node.position[1] + new_position[1],
                current_node.position[2] + new_position[2],
            )

            # Make sure within range
            if (
                node_position[0] > (maze.shape[0] - 1)
                or node_position[0] < 0
                or node_position[1] > (maze.shape[1] - 1)
                or node_position[1] < 0
            ):
                continue

            # Make sure walkable terrain
            if maze[node_position[0], node_position[1], node_position[2]] != 0:
                continue

            # Create new node
            new_node = Node(
                current_node,
                node_position,
                delta(new_position),
            )

            # Append
            children.append(new_node)

        # Loop through children
        for child in children:

            # Child is on the closed list
            for closed_child in closed_list:
                if child == closed_child:
                    continue

            # Create the f, g, and h values
            child.g = current_node.g + child.delta
            child.h = heuristic(child)

            # Child is already in the open list
            for open_node in open_list:
                if child == open_node and child.g > open_node.g:
                    continue

            # Add the child to the open list
            heapq.heappush(open_list, child)


def main():
    maze = np.array(
        [
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        ],
        np.int8,
    )

    start = (0, 0, 0)
    end = (0, 8, 3)

    path = astar(maze, start, end)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for z in range(0, maze.shape[0])
        for y in range(0, maze.shape[1]):
            for x in range(0, maze.shape[2]):
                if maze[[z], [y], [x]] != 0:
                    ax.scatter(x , y, z, marker="o", markerfacecolor="blue",)

    for point in path:
        ax.scatter(point[0] , point[1], point[2], marker="o", markerfacecolor="magenta",)
        
    ax.scatter(start[0], start[1], start[2], marker="o", markerfacecolor="red",)
    ax.scatter(end[0], end[1], end[2], marker="o", markerfacecolor="green",)

    plt.show()


if __name__ == "__main__":
    main()

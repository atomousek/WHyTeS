import networkx
import numpy as np
import math


class PathFinder:

    def __init__(self, path, edges_of_cell):
        #self.data = np.loadtxt(path)
        data = np.loadtxt(path)
        self.edges_of_cell = edges_of_cell
        self.graph = networkx.DiGraph()
        self.shortest_path = None
        self.trajectory = None
        self.intersections = []
        self.x_max = np.max(data[:, 1])
        self.x_min = np.min(data[:, 1])
        self.y_max = np.max(data[:, 2])
        self.y_min = np.min(data[:, 2])
        self.x_range = self.x_max - self.x_min
        self.y_range = self.y_max - self.y_min
        self.shape = (int(self.y_range / edges_of_cell[2]) + 1, int(self.x_range / edges_of_cell[1]) + 1)

    def get_real_coordinate(self, row, column):
        """
        :param row:
        :param column:
        :param x_min:
        :param y_max:
        :param edges_of_cell:
        :return: (x, y) coordiates
        """
        x_length = edges_of_cell[1]
        y_length = edges_of_cell[2]

        return [self.x_min + x_length*column - x_length*0.5,  self.y_max - y_length*row + y_length*0.5]

    def get_index(self, x, y):
        """
        :param x:
        :param y:
        :return: (row, column) index
        """
        x_length = edges_of_cell[1]
        y_length = edges_of_cell[2]

        return  [int((self.y_max-y)/y_length),  int((x - self.x_min)/x_length)]

    def get_weight(self):


        return 1

    def fill_grid_with_data(self, data, grid):

        for i in xrange(len(data[:, 0])):
            row, column = self.get_index(data[i, 1], data[i, 2])
            grid[row, column] = data[i, 3]

        return grid

    def creat_graph(self):

        shape = self.shape
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                src = (i, j)
                # if the cell is not on border
                if i != 0 and j != 0 and i != shape[0]-1 and j != shape[1]-1:
                    # upper cell
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower cell
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # right cell
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # left cell
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                    # upper right cell
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower right cell
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper left cell
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower left cell
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif i == 0 and j == shape[1]-1:    # top right corner
                    # left cell
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower cell
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower left cell
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif i == 0 and j == 0:    # top left corner
                    # right cell
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower cell
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower right cell
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif i == shape[0]-1 and j == shape[1]-1:    # bottom right corner
                    # upper cell
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # left cell
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper left cell
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif i == shape[0]-1 and j == 0:    # bottom left corner
                    # upper cell
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # right cell
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper right cell
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif i == 0:    # top border
                    # left cell
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # right cell
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower cell
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower left cell
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower right cell
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif i == shape[0]-1:    # bottom border
                    # left cell
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # right cell
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper cell
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper right cell
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper left cell
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif j == shape[1]-1:   # right border
                    # left cell
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper cell
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower cell
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper left cell
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower left cell
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

                elif j == 0:    # left border
                    # right cell
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper cell
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower cell
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # upper right cell
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())
                    # lower right cell
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight())

        return

    def remove_walls(self, walls):

        for i in xrange(walls.shape[0]):
            if 1.0 < walls[i][2]:
                row, column = self.get_index(walls[i][0], walls[i][1])
                self.graph.remove_node((row, column))

        return

    def extract_path(self):
        positions = np.zeros((len(self.shortest_path), 2))

        for i in xrange(len(self.shortest_path)):
            x, y = self.get_real_coordinate(self.shortest_path[i][0], self.shortest_path[i][1])
            positions[i, 0] = x
            positions[i, 1] = y
            print str(self.shortest_path[i]) + "   " + str(x) + "    " + str(y)

        np.savetxt('../results/path.txt', positions)
        return positions

    def extract_trajectory(self, path, start_time, speed):
        times = np.zeros(path.shape[0])
        labels = np.ones(path.shape[0])
        labels[(labels == 1)] = 2
        times[0] = start_time
        time = start_time
        for i in xrange(1, path.shape[0]):
            distance = math.sqrt(((path[i-1, 0] - path[i, 0]) ** 2) + ((path[i-1, 1] - path[i, 1]) ** 2))
            time = time + distance/speed
            times[i] = time

        out = np.c_[times, path, labels]
        np.savetxt('../results/trajectory.txt', out)
        self.trajectory = out
        return out

    def find_shortest_path(self, src, dst):

        self.shortest_path = list(networkx.shortest_path(self.graph, src, dst))
        return self.shortest_path

    def _round_time(self, data):
        for i in xrange(len(data[:, 0])):
            data[i, 0] = int(data[i, 0])
        return data

    def calculate_intersections(self, path):
        data = np.loadtxt(path)
        data = self._round_time(data)
        k = 0
        for i in xrange(len(self.trajectory[:, 0])):
            row_robot, column_robot = self.get_index(self.trajectory[i, 1], self.trajectory[k, 2])
            while data[k, 0] ==  self.trajectory[i, 0]:
                row_data, column_data = self.get_index(data[k, 1], data[k, 2])
                if row_data == row_robot and column_data == column_robot:
                    self.intersections.append((data[k, 1], data[k, 2]))

        return self.intersections


if __name__ == "__main__":

    edges_of_cell = np.array([3600., 1., 1.])
    path_finder = PathFinder('../results/outliers.txt', edges_of_cell)
    path_finder.creat_graph()
    #walls = np.loadtxt('../data/walls.txt')
    #path_finder.remove_walls(walls)
    print path_finder.find_shortest_path((0, 0), (15, 13))
    path_finder.extract_path()
    p = np.loadtxt('../results/path.txt')
    path_finder.extract_trajectory(p, 100, 1)

    #print path_finder.calculate_intersections('../results/path.txt')

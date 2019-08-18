import networkx
import numpy as np
import math


class PathFinder:

    def __init__(self, path, edges_of_cell):
        self.data = np.loadtxt(path)
        # data = np.loadtxt(path)
        self.edges_of_cell = edges_of_cell
        self.graph = networkx.DiGraph()
        self.shortest_path = []
        self.trajectory = None
        self.intersections = []
        self.x_min = -9.25
        self.x_max = 3.0
        self.y_min = 0.0
        self.y_max = 16.0

        # self.x_max = np.max(data[:, 1])
        # self.x_min = np.min(data[:, 1])
        # self.y_max = np.max(data[:, 2])
        # self.y_min = np.min(data[:, 2])
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
        x_length = self.edges_of_cell[1]
        y_length = self.edges_of_cell[2]

        return (self.x_min + x_length*column - x_length*0.5,  self.y_max - y_length*row + y_length*0.5)

    def get_index(self, x, y):
        """
        :param x:
        :param y:
        :return: (row, column) index
        """
        x_length = self.edges_of_cell[1]
        y_length = self.edges_of_cell[2]

        return  (int((self.y_max-y)/y_length+1),  int((x - self.x_min)/x_length+1))

    def get_weight(self, dst, angle):
        real_dst = self.get_real_coordinate(dst[0], dst[1])
        k = 0
        while k < len(self.data[:, 0]):
            if self.data[k, 1] == real_dst[0] and self.data[k, 2] == real_dst[1] and self.data[k, 3] == angle:
                return self.data[k, 4]
            k += 1

        return 10000000

    def fill_grid_with_data(self, data, grid):

        for i in xrange(len(data[:, 0])):
            row, column = self.get_index(data[i, 1], data[i, 2])
            grid[row, column] = data[i, 3]

        return grid

    def creat_graph(self):
        directions = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4]
        shape = self.shape
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                src = (i, j)
                # if the cell is not on border
                if i != 0 and j != 0 and i != shape[0]-1 and j != shape[1]-1:
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6]))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2]))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4]))
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0]))

                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5]))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3]))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7]))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1]))

                elif i == 0 and j == shape[1]-1:    # top right corner
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0]))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2]))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1]))

                elif i == 0 and j == 0:    # top left corner
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4]))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2]))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3]))

                elif i == shape[0]-1 and j == shape[1]-1:    # bottom right corner
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6]))
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0]))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7]))

                elif i == shape[0]-1 and j == 0:    # bottom left corner
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6]))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4]))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5]))

                elif i == 0:    # top border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0]))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4]))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2]))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1]))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3]))

                elif i == shape[0]-1:    # bottom border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0]))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4]))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6]))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5]))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7]))

                elif j == shape[1]-1:   # right border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0]))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6]))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2]))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7]))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1]))

                elif j == 0:    # left border
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4]))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6]))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2]))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5]))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3]))

        return

    def remove_walls(self, walls):

        for i in xrange(walls.shape[0]):
            row, column = self.get_index(walls[i][0], walls[i][1])
            try:
                self.graph.remove_node((row, column))

            except:
                pass
        return

    def extract_path(self):
        positions = np.zeros((len(self.shortest_path), 2))

        for i in xrange(len(self.shortest_path)):
            x, y = self.get_real_coordinate(self.shortest_path[i][0], self.shortest_path[i][1])
            positions[i, 0] = x
            positions[i, 1] = y

        np.savetxt('../results/path.txt', positions)
        return positions

    def extract_trajectory(self, path, start_time, speed):
        times = np.zeros(path.shape[0])
        labels = np.full(path.shape[0], 2)
        # labels[(labels == 1)] = 2
        times[0] = start_time
        time = start_time
        for i in xrange(1, path.shape[0]):
            if path[i-1, 0] == path[i, 0] and path[i-1, 1] == path[i, 1]:
                pass
            distance = math.sqrt(((path[i-1, 0] - path[i, 0]) ** 2) + ((path[i-1, 1] - path[i, 1]) ** 2))
            time = time + distance/speed
            times[i] = time

        out = np.c_[times, path, labels, labels, labels]
        np.savetxt('../results/trajectory.txt', out)
        self.trajectory = out
        return out

    def find_shortest_path(self, route):
        # src_index = self.get_index(src[0], src[1])
        # dst_index = self.get_index(dst[0], dst[1])

        try:
            for i in xrange(1, len(route)):
                src = self.get_index(route[i-1][0], route[i-1][1])
                dst = self.get_index(route[i][0], route[i][1])
                # self.shortest_path.extend(list(networkx.shortest_path(self.graph, src, dst)))
                self.shortest_path.extend(list(networkx.shortest_path(self.graph, src, dst)))
        except:
            print "no path!"
        return self.shortest_path

    def _round_time(self, data):
        for i in xrange(len(data[:, 0])):
            data[i, 0] = int(data[i, 0])
        return data

    def extract_intersections(self, path):
        data = np.loadtxt(path)
        data = data[data[:, 0].argsort()]
        data = self._round_time(data)
        k = 0
        for i in xrange(len(self.trajectory[:, 0])):
            row_robot, column_robot = self.get_index(self.trajectory[i, 1], self.trajectory[i, 2])
            while int(data[k, 0]) ==  int(self.trajectory[i, 0]):
                row_data, column_data = self.get_index(data[k, 1], data[k, 2])
                if row_data == row_robot and column_data == column_robot:
                    self.intersections.append((data[k, 0], data[k, 1], data[k, 2], 0, 0, 3))
                k += 1

        np.savetxt('../results/intersections.txt', np.array(self.intersections))
        return self.intersections

    def extract_centers(self):
        directions = [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi, -3*np.pi/4, -np.pi/2, -np.pi/4]
        speeds = [0.5, 1.5, 2.5, 3.5]
        k = 0
        centers = np.zeros((self.shape[0]*self.shape[1]*8*4, 4))
        for j in xrange(self.shape[1]):
            for i in xrange(self.shape[0]):
                x, y = self.get_real_coordinate(i, j)
                for z in xrange(8):
                    for s in xrange(len(speeds)):
                        centers[k, 0] = x
                        centers[k, 1] = y
                        centers[k, 2] = directions[z]
                        centers[k, 3] = speeds[s]
                        k+=1


        np.savetxt('../results/centers.txt', centers)
if __name__ == "__main__":

    edges_of_cell = np.array([3600., 0.5, 0.5])
    path_finder = PathFinder('../data/model_of_8angles_0.5m.txt', edges_of_cell)
    path_finder.creat_graph()
    # path_finder.extract_centers()

    walls = np.loadtxt('../data/artificial_boarders_of_space_in_UTBM.txt')
    path_finder.remove_walls(walls)
    # route = [(-8.5, 15), (2, 15), (2, 3), (-8.5, 15), (-8.5, 3), (2, 15)]
    route = [(-3.4, 13.2), (2, 15), (2, 3), (-8.5, 15), (-8.5, 3), (2, 15)]
    path_finder.find_shortest_path(route)
    path_finder.extract_path()
    p = np.loadtxt('../results/path.txt')
    testing_data_path = '../data/testing_data.txt'
    data = np.loadtxt(testing_data_path)
    path_finder.extract_trajectory(p, np.min(data[:, 0]), speed = 0.5)
    # print path_finder.graph.nodes

    print path_finder.extract_intersections(testing_data_path)

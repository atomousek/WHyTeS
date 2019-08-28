import networkx
from networkx.exception import NetworkXError
import numpy as np
import math


class PathFinder:

    def __init__(self, path, edges_of_cell):
        self.data = np.loadtxt(path)
        # data = np.loadtxt(path)
        self.edges_of_cell = edges_of_cell
        self.graph = networkx.DiGraph()
        self.shortest_path = []
        # self.path_weights = []
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

        return (self.x_min + x_length*column - x_length*0.5, self.y_max - y_length*row + y_length*0.5)

    def get_index(self, x, y):
        """
        :param x:
        :param y:
        :return: (row, column) index
        """
        x_length = self.edges_of_cell[1]
        y_length = self.edges_of_cell[2]

        return  (int((self.y_max-y)/y_length+1), int((x - self.x_min)/x_length+1))

    def get_weight(self, dst, angle, dummy=False):
        if dummy:
            return 1.
        
        real_dst = self.get_real_coordinate(dst[0], dst[1])
        k = 0
        
        # index = np.where(data[:, 1] == real_dst[0])
        while k < len(self.data[:, 0]):
            if self.data[k, 0] == real_dst[0] and self.data[k, 1] == real_dst[1] and self.data[k, 2] == angle:
                return self.data[k, 3]
            k += 1

        return 10000

    def fill_grid_with_data(self, data, grid):

        for i in xrange(len(data[:, 0])):
            row, column = self.get_index(data[i, 1], data[i, 2])
            grid[row, column] = data[i, 3]

        return grid

    def creat_graph(self, dummy=False):
        self.graph = networkx.DiGraph()
        directions = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi, -3 * np.pi / 4, -np.pi / 2, -np.pi / 4]
        shape = self.shape
        for i in xrange(shape[0]):
            for j in xrange(shape[1]):
                src = (i, j)
                # if the cell is not on border
                if i != 0 and j != 0 and i != shape[0]-1 and j != shape[1]-1:
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))

                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))

                elif i == 0 and j == shape[1]-1:    # top right corner
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))

                elif i == 0 and j == 0:    # top left corner
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))

                elif i == shape[0]-1 and j == shape[1]-1:    # bottom right corner
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))

                elif i == shape[0]-1 and j == 0:    # bottom left corner
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))

                elif i == 0:    # top border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))

                elif i == shape[0]-1:    # bottom border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))

                elif j == shape[1]-1:   # right border
                    # left neighbor
                    dst = (i, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[0], dummy))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # upper left neighbor
                    dst = (i - 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[7], dummy))
                    # lower left neighbor
                    dst = (i + 1, j - 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[1], dummy))

                elif j == 0:    # left border
                    # right neighbor
                    dst = (i, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[4], dummy))
                    # upper neighbor
                    dst = (i - 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[6], dummy))
                    # lower neighbor
                    dst = (i + 1, j)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[2], dummy))
                    # upper right neighbor
                    dst = (i - 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[5], dummy))
                    # lower right neighbor
                    dst = (i + 1, j + 1)
                    self.graph.add_edge(src, dst, weight=self.get_weight(dst, directions[3], dummy))

        return

    def remove_walls(self, walls):

        for position in walls:
            row, column = self.get_index(position[0], position[1])
            try:
                self.graph.remove_node((row, column))
            except NetworkXError:
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
        self.trajectory = None
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

        out = np.c_[times, path, labels, labels]
        np.savetxt('../results/trajectory.txt', out)
        self.trajectory = out
        return out

    def find_shortest_path(self, route):
        self.shortest_path = []
        
            
        try:
            for i in xrange(1, len(route)):
                src = self.get_index(route[i-1][0], route[i-1][1])
                dst = self.get_index(route[i][0], route[i][1])
                # self.shortest_path.extend(list(networkx.shortest_path(self.graph, src, dst)))
                self.shortest_path.extend(list(networkx.dijkstra_path(self.graph, src, dst)))
                # self.path_weights.append(networkx.shortest_path_length(self.graph, src, dst))
        except networkx.NetworkXError:
            print "no path!"

        return self.shortest_path

    def _round_time(self, data):
        for line in data:
            line[0] = int(line[0])
        return data

    def extract_intersections(self, path, radius):
        self.intersections = []
        data = np.loadtxt(path)
        data = data[data[:, 0].argsort()]
        # np.savetxt('../data/test_dataset_all.txt', data)
        # data = self._round_time(data)
        k = 0
        counter = 0

        for position in self.trajectory:
            x_robot = position[1]
            y_robot = position[2]

            mask = np.where((position[0] - 0.5 < data[:, 0]) & (position[0] + 0.5 > data[:, 0]))


            for index in mask[0]:
                x_pedestrian = data[index, 1]
                y_pedestrian = data[index, 2]
                dist = ((x_robot - x_pedestrian) ** 2 + (y_robot - y_pedestrian) ** 2) ** 0.5
                if dist <= radius:
                    self.intersections.append((data[index, 0], data[index, 1], data[index, 2],  data[index, 7], 3))
                    counter += 1


            # if k < len(data[:, 0]) and data[k, 0] > position[0]:
            #   pass
            #
            # while k < len(data[:, 0]) and 0 <= position[0] - data[k, 0] < 1:
            #     x_pedestrian = data[k, 1]
            #     y_pedestrian = data[k, 2]
            #     dist = ((x_robot - x_pedestrian) ** 2 + (y_robot - y_pedestrian) ** 2) ** 0.5
            #     if dist <= radius:
            #         self.intersections.append((data[k, 0], data[k, 1], data[k, 2],  data[k, 7], 3))
            #         counter += 1
            #     k += 1
            #
            # m = k + 1
            #
            # while m < len(data[:, 0]) and 0 <= data[m, 0] - position[0] < 1:
            #     x_pedestrian = data[m, 1]
            #     y_pedestrian = data[m, 2]
            #     dist = ((x_robot - x_pedestrian) ** 2 + (y_robot - y_pedestrian) ** 2) ** 0.5
            #     if dist <= radius:
            #         self.intersections.append((data[m, 0], data[m, 1], data[m, 2], data[m, 7], 3))
            #         counter += 1
            #     m += 1

        np.savetxt('../results/intersections.txt', np.array(self.intersections))

        if counter > 0:
            return counter
            return len(np.unique(np.array(self.intersections)[:, 3]))
        else:
            return 0

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

    def get_path_weight(self):
        total_weight = 0
        for i in xrange(len(self.shortest_path)-1):
            weight = self.graph.get_edge_data(self.shortest_path[i], self.shortest_path[i+1])
            if weight != None:
                total_weight += weight['weight']

        return total_weight

    def find_strange_path(self, point_1, point_2, point_3):

        ## NOT WORKING

        route_1 = [point_1, point_2, point_3, point_1]
        route_2 = [point_1, point_3, point_2, point_1]
        path_1 = []
        path_2 = []

        try:
            for i in xrange(1, len(route_1)):
                src_1 = self.get_index(route_1[i-1][0], route_1[i-1][1])
                src_2 = self.get_index(route_2[i-1][0], route_2[i-1][1])
                dst_1 = self.get_index(route_1[i][0], route_1[i][1])
                dst_2 = self.get_index(route_2[i][0], route_2[i][1])
                path_1.extend(list(networkx.shortest_path(self.graph, src_1, dst_1)))
                path_2.extend(list(networkx.shortest_path(self.graph, src_2, dst_2)))
        except:
            print "no path!"

        weight_1 = self._get_path_weight(path_1)
        weight_2 = self._get_path_weight(path_2)

        if weight_1 < weight_2:
            self.shortest_path = path_1
            return path_1
        else:
            self.shortest_path = path_2
            return path_2



if __name__ == "__main__":

    edges_of_cell = np.array([3600., 0.5, 0.5])
    path_finder = PathFinder('../data/1554139930_model.txt', edges_of_cell)
    path_finder.creat_graph(True)
    walls = np.loadtxt('../data/artificial_boarders_of_space_in_UTBM.txt')
    path_finder.remove_walls(walls)

    route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]
    path_finder.find_shortest_path(route)
    # path_finder.find_strange_path((-5, 10), (1, 10), (1, 2))
    path_finder.extract_path()
    path = np.loadtxt('../results/path.txt')
    testing_data_path = '../data/1554139930_test_data.txt'

    data = np.loadtxt(testing_data_path)
    path_finder.extract_trajectory(path, np.min(data[:, 0]), speed=1)
    
    print path_finder.extract_intersections(testing_data_path, 1.)

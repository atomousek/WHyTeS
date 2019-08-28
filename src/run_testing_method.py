import numpy as np
import path_finder as pf


class Tester:

    def __init__(self):

        return

    def test_model(self, path_model, path_data, edges_of_cell, speed):
        results = []
        size_of_robot = 1.
        min_time = np.min(np.loadtxt(path_data)[:, 0])

        results.append(int(min_time))

        path_finder = pf.PathFinder(path=path_model, edges_of_cell=edges_of_cell)

        path_finder.creat_graph()
        walls = np.loadtxt('../data/artificial_boarders_of_space_in_UTBM.txt')
        path_finder.remove_walls(walls)

        route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]
        reverse_route = [(-5, 10), (-7, 1), (2, 3), (-5, 10)]
        path_finder.find_shortest_path(route=route)
        path_finder.extract_path()
        robot_path = np.loadtxt('../results/path.txt')

        path_finder.extract_trajectory(robot_path, min_time, speed=speed)
        results.append(path_finder.extract_intersections(path_data, radius=size_of_robot))

        path_finder.find_shortest_path(route=reverse_route)
        path_finder.extract_path()
        robot_path = np.loadtxt('../results/path.txt')
        path_finder.extract_trajectory(robot_path, min_time, speed=speed)
        results.append(path_finder.extract_intersections(path_data, radius=size_of_robot))

        return results


if __name__ == "__main__":

    tester = Tester()
    edges_of_cell = [3600., 0.5, 0.5]
    print tester.test_model('../data/1554139930_model.txt', '../data/1554139930_test_data.txt', edges_of_cell, speed=1.0)
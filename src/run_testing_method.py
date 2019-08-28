import numpy as np
import path_finder as pf


class Tester:

    def __init__(self, radius_of_robot):
        self.radius_of_robot = radius_of_robot


    def test_model(self, path_model, path_data, edges_of_cell, speed):
        results = []
        test_data = np.loadtxt(path_data)
        min_time = np.min(test_data[:, 0])
        number_of_detections = len(test_data[:, 0])
        route = [(-5, 10), (2, 3), (-7, 1), (-5, 10)]
        reverse_route = [(-5, 10), (-7, 1), (2, 3), (-5, 10)]

        results.append(int(min_time))
        results.append(number_of_detections)

        path_finder = pf.PathFinder(path=path_model, edges_of_cell=edges_of_cell)



        '''
        Dummy Model
        '''
        path_finder.creat_graph(dummy=True)
        walls = np.loadtxt('../data/artificial_boarders_of_space_in_UTBM.txt')
        path_finder.remove_walls(walls)

        path_finder.find_shortest_path(route=route)
        path_finder.extract_path()
        robot_path = np.loadtxt('../results/path.txt')

        path_finder.extract_trajectory(robot_path, min_time, speed=speed)
        results.append(path_finder.extract_intersections(path_data, radius=self.radius_of_robot))

        path_finder.find_shortest_path(route=reverse_route)
        path_finder.extract_path()
        robot_path = np.loadtxt('../results/path.txt')
        path_finder.extract_trajectory(robot_path, min_time, speed=speed)
        results.append(path_finder.extract_intersections(path_data, radius=self.radius_of_robot))


        '''
        Real Model
        '''
        path_finder.creat_graph()
        walls = np.loadtxt('../data/artificial_boarders_of_space_in_UTBM.txt')
        path_finder.remove_walls(walls)

        path_finder.find_shortest_path(route=route)
        path_finder.extract_path()
        weight_1 = path_finder.get_path_weight()
        robot_path = np.loadtxt('../results/path.txt')

        path_finder.extract_trajectory(robot_path, min_time, speed=speed)
        results.append(path_finder.extract_intersections(path_data, radius=self.radius_of_robot))

        path_finder.find_shortest_path(route=reverse_route)
        path_finder.extract_path()
        weight_2 = path_finder.get_path_weight()
        robot_path = np.loadtxt('../results/path.txt')
        path_finder.extract_trajectory(robot_path, min_time, speed=speed)
        results.append(path_finder.extract_intersections(path_data, radius=self.radius_of_robot))

        # if weight_1 < weight_2:
        #     results.append(1)
        # else:
        #     results.append(2)
        #

        results.append(weight_1)
        results.append(weight_2)
        return results


if __name__ == "__main__":

    tester = Tester(radius_of_robot=1.)
    edges_of_cell = [3600., 0.5, 0.5]
    print tester.test_model('../data/1554105948_model.txt', '../data/1554105948_test_data.txt', edges_of_cell, speed=1.0)
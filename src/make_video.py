import numpy as np
import cv2
from datetime import datetime


class VideoMaker:

    def __init__(self, path_data, path_trajectory, path_intersections, path_borders, edges_of_cell, size_factor):
        self.data = np.loadtxt(path_data)
        self.trajectory = np.loadtxt(path_trajectory)
        self.intersections = np.loadtxt(path_intersections).reshape((1, self.data.shape[1]))
        self.edges_of_cell = edges_of_cell
        self.path_borders = path_borders
        # self.x_max = np.max(self.data[:, 1])
        # self.x_min = np.min(self.data[:, 1])
        # self.y_max = np.max(self.data[:, 2])
        # self.y_min = np.min(self.data[:, 2])
        self.x_min = -9.25
        self.x_max = 3.0
        self.y_min = 0.0
        self.y_max = 16.0
        self.time_min = int(np.min(self.data[:, 0]))
        self.time_max = int(np.max(self.data[:, 0]))
        self.size_factor = size_factor
        self.shape = (int((self.y_max - self.y_min) * size_factor / edges_of_cell[2]),
                     int((self.x_max - self.x_min) * size_factor / edges_of_cell[1]))

    def _draw_grid(self, frame, line_color=(0, 255, 0), thickness=1, type=cv2.LINE_AA):

        for i in xrange(int(self.x_min), int(self.x_max), 1):
            x, y = self.get_frame_index(i, self.y_min)
            cv2.line(frame, (x, 0), (x, y), color=line_color, lineType=type, thickness=thickness)

        for i in xrange(int(self.y_min), int(self.y_max), 1):
            x, y = self.get_frame_index(self.x_max, i)
            cv2.line(frame, (x, y), (0, y), color=line_color, lineType=type, thickness=thickness)

        return

    def get_frame_index(self, x, y):

        x_length = self.edges_of_cell[1]
        y_length = self.edges_of_cell[2]

        return (int((x - self.x_min) * self.size_factor / x_length), int((self.y_max - y) * self.size_factor / y_length))

    def _create_empty_frame(self):
        frame = cv2.cvtColor(np.full(self.shape, 255, np.uint8), cv2.COLOR_GRAY2BGR)
        self._draw_grid(frame, (150, 150, 150), 1, cv2.LINE_4)
        borders = np.loadtxt(self.path_borders)
        for i in xrange(len(borders[:, 0])):
            pos = self.get_frame_index(borders[i, 0], borders[i, 1])
            cv2.circle(frame, pos, int(0.2 * edges_of_cell[2] * self.size_factor),
                       (0, 0, 0), int(0.1 * self.size_factor))

        return frame

    def make_video (self, fps=20):
        concatenated = np.concatenate((self.data, self.trajectory, self.intersections), axis=0)
        concatenated = concatenated[concatenated[:, 0].argsort()]

        times = np.arange(self.time_min, self.time_max, 1)
        counter = 0
        k = 0
        number_of_frames = len(times)

        # name = '../results/%sx%sx%s.avi' % (str(int(edges_of_cell[0])), str(edges_of_cell[1]), str(edges_of_cell[2]))
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        out = cv2.VideoWriter('../results/trajectory.avi', fourcc, fps, (self.shape[1], self.shape[0]))

        pedestrians = []
        empty_frame = self._create_empty_frame()
        last_pos_robot = (0, 0)
        for time in times:

            frame = empty_frame.copy()
            name = str(time)
            del pedestrians[:]
            while k < len(concatenated[:, 0]) and time == int(concatenated[k, 0]):

                # labeling the robot
                if int(concatenated[k, 5]) == 2:
                    last_pos_robot = self.get_frame_index(concatenated[k, 1], concatenated[k, 2])

                # labeling the pedestrians
                if int(concatenated[k, 5]) == 1:
                    pos = self.get_frame_index(concatenated[k, 1], concatenated[k, 2])
                    if len(pedestrians) == 0:
                        pedestrians.append((pos, concatenated[k, 3]))
                    for j in xrange(len(pedestrians)):
                        if pedestrians[j][1] == concatenated[k, 3]:
                            del pedestrians[j]
                            pedestrians.append((pos, concatenated[k, 3]))
                        elif j == len(pedestrians):
                            pedestrians.append((pos, concatenated[k, 3]))
                    # cv2.circle(frame, pos, int(0.5 * edges_of_cell[2] * self.size_factor),
                    #            (250, 0, 0), int(0.2 * self.size_factor))
                    # cv2.putText(frame, str(int(concatenated[k, 3]*100)), pos, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)

                # labeling the intersections
                if int(concatenated[k, 5]) == 3:
                    pos = self.get_frame_index(concatenated[k, 1], concatenated[k, 2])
                    cv2.circle(frame, pos, int(2 * edges_of_cell[2] * self.size_factor),
                               (0, 0, 250), int(0.1 * self.size_factor))

                k += 1

            for pedestrian in pedestrians:
                cv2.circle(frame, pedestrian[0], int(0.5 * edges_of_cell[2] * self.size_factor),
                           (250, 0, 0), int(0.2 * self.size_factor))
                cv2.putText(frame, str(int(pedestrian[1] * 100)), pedestrian[0], cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255),
                            lineType=cv2.LINE_4)
            cv2.circle(frame, last_pos_robot, int(0.5 * edges_of_cell[2] * self.size_factor),
                       (0, 250, 0), int(0.5 * self.size_factor))
            cv2.putText(frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)
            # cv2.putText(frame, datetime.utcfromtimestamp(times[i]).strftime('Day:%d, Time:%H:%M:%S'), (9*self.size_factor, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.06*self.size_factor, (0, 0, 255), lineType=cv2.LINE_4)
            counter += 1
            out.write(frame)

        cv2.destroyAllWindows()
        print 'saving video..'
        out.release()

        return counter


if __name__ == "__main__":

    path_data = '../data/testing_data.txt'
    path_trajectory = '../results/trajectory.txt'
    path_borders = '../data/artificial_boarders_of_space_in_UTBM.txt'
    path_intersections = '../results/intersections.txt'
    edges_of_cell=np.array([1.0, 0.5, 0.5])
    vm = VideoMaker(path_data, path_trajectory, path_intersections, path_borders, edges_of_cell, size_factor=20)

    print vm.make_video(fps=5)

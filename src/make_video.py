import numpy as np
import cv2
from datetime import datetime


# This function taken from https://stackoverflow.com/questions/44816682/drawing-grid-lines-across-the-image-uisng-opencv-python
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep


def make_video (path_data, path_outliers, edges_of_cell=np.array([3600.0, 1.0, 1.0]), size_factor=10, fps=20):
    '''
        Params:
            path_data:          path to a numpy matrix with following columns; (timestamp(float), x_coordinate(float),
                                                y_coordinate(float), 1.0)   last column can be removed in the future
            path_outliers:      path to outlers.txt, which created by frequencies.py
            edges_of_cell:      Cell size used in method
            size_factor:        this specifies the frame size as ( x_range * size_factor ,  y_range * size_factor )
            fps:                frames per second

        Return: total number of frames
    '''

    data = np.loadtxt(path_data)
    outliers = np.loadtxt(path_outliers)
    cell_time = edges_of_cell[0]

    for i in range(len(outliers[:, 0]), 0, -1):
        #print i
        if outliers[i-1, 3] == 0:
            outliers = np.delete(outliers, i-1, 0)
        elif outliers[i-1, 3] == 2.0 or outliers[i-1, 3] == -1.0:
            outliers[i - 1, 0] = outliers[i - 1, 0] - cell_time / 2

    concatenated = np.concatenate((data, outliers), axis=0)
    concatenated = concatenated[concatenated[:, 0].argsort()]

    times = np.unique(concatenated[:, 0])
    x_max = np.max(data[:, 1])
    x_min = np.min(data[:, 1])
    y_max = np.max(data[:, 2])
    y_min = np.min(data[:, 2])
    counter = 0
    k = 0
    last_outlier = 0
    last_time = cell_time

    width = int((x_max-x_min)*size_factor)
    height = int((y_max-y_min)*size_factor)
    number_of_frames = len(times)

    name = '../results/%sx%sx%s.avi' % (str(int(edges_of_cell[0])), str(edges_of_cell[1]), str(edges_of_cell[2]))
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(name, fourcc, fps, (width, height))

    # initilazing the empty frame
    empty_frame = cv2.cvtColor(np.full([height, width], 255, np.uint8), cv2.COLOR_GRAY2BGR)
    draw_grid(empty_frame, (150, 150, 150), 1, cv2.LINE_4, size_factor)
    outlier_frame = empty_frame.copy()

    for i in xrange(number_of_frames):
        # if times[i] >= last_outlier + cell_time:
        #
        #     if last_time < last_outlier + cell_time:
        #         while last_time < last_outlier + cell_time:
        #             temp = outlier_frame.copy()
        #             last_time += 0.5
        #             cv2.putText(temp, str(last_time), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)
        #             cv2.putText(temp, datetime.utcfromtimestamp(last_time).strftime('Day:%d, Time:%H:%M:%S'),
        #                         (9 * size_factor, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.06 * size_factor, (0, 0, 255),
        #                         lineType=cv2.LINE_4)
        #             counter += 1
        #             out.write(temp)
        #
        #     # print "FLUSH!"
        #     frame = empty_frame.copy()
        #     outlier_frame = empty_frame.copy()
        # else:
        #     # print "keep going.."
        #     frame = outlier_frame.copy()
        frame = empty_frame.copy()
        name = str(times[i])
        while k < len(concatenated[:, 0]) and times[i] == concatenated[k, 0]:

            # labeling the robot
            if concatenated[k, 3] == 2.0:
                cv2.circle(outlier_frame, (int(concatenated[k, 1] + abs(x_min)) * size_factor,
                                           height - int(concatenated[k, 2] + abs(y_min)) * size_factor), int(0.7 * edges_of_cell[2] *size_factor),
                           (250, 250, 0), int(0.1 * size_factor))
                last_outlier = concatenated[k, 0]

            # labeling the occurrences
            if concatenated[k, 3] == 1:

                cv2.circle(frame, (int(concatenated[k, 1] + abs(x_min))*size_factor,
                                   height - int(concatenated[k, 2] + abs(y_min))*size_factor),
                           int(0.2*size_factor), (255, 0, 0), int(0.1*size_factor))

            k += 1
        cv2.putText(frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)
        cv2.putText(frame, datetime.utcfromtimestamp(times[i]).strftime('Day:%d, Time:%H:%M:%S'),
                    (9*size_factor, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.06*size_factor, (0, 0, 255), lineType=cv2.LINE_4)
        counter += 1
        last_time = times[i]
        out.write(frame)

    cv2.destroyAllWindows()
    print 'saving video..'
    out.release()

    return counter

path_data = '../data/data_for_visualization/wednesday_thursday_days_nights_only_ones.txt'
path_trajectory = '../results/trajectory.txt'

edges_of_cell=np.array([600.0, 0.5, 0.5])
print make_video(path_data, path_trajectory, edges_of_cell=edges_of_cell)
import numpy as np
import cv2


def make_video (data):
    times = np.unique(data[:, 0])
    x_max = np.max(data[:, 1])
    x_min = np.min(data[:, 1])
    y_max = np.max(data[:, 2])
    y_min = np.min(data[:, 2])
    width = int((x_max-x_min)*10)
    height = int((y_max-y_min)*10)
    number_of_frames = len(times)

    k = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('people.avi', fourcc, 1, (width, height))
    for i in xrange(number_of_frames):
        frame = cv2.cvtColor(np.full([height, width], 255, np.uint8), cv2.COLOR_GRAY2BGR)
        name = str(times[i])
        counter = 0
        while k < len(data[:, 0]) and times[i] == data[k, 0]:
            if data[k, 3] == 0:
                cv2.circle(frame, (int(data[k, 2] + abs(y_min))*10, int(data[k, 1] + abs(x_min))*10), 2, (255, 0, 0), 1)
                counter += 1
            k += 1
        #cv2.circle(frame, (frame,  150), 5, (0, 0, 255), 5)
        #out.write(frame)

        if counter > 5:
            filename = "%s.png" % name
            #cv2.imwrite(filename, frame)
            out.write(frame)

    cv2.destroyAllWindows()
    out.release()

    return 0

path = '../data/two_weeks_days_nights_weekends.txt'

dataset = np.loadtxt(path)
dataset = dataset[0:439724]

make_video(dataset)
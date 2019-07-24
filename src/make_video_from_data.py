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
    out = cv2.VideoWriter('people.avi', fourcc, 20, (width, height))
    for i in xrange(number_of_frames):
        frame = cv2.cvtColor(np.full([height, width], 255, np.uint8), cv2.COLOR_GRAY2BGR)
        name = str(times[i])
        counter = 0
        while k < len(data[:, 0]) and times[i] == data[k, 0]:
            if data[k, 3] == 1:
                cv2.circle(frame, (int(data[k, 2] + abs(y_min))*10, int(data[k, 1] + abs(x_min))*10), 2, (255, 0, 0), 1)
                counter += 1
            k += 1
        frame = cv2.putText(frame, name, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_4)
        out.write(frame)

        #if counter > 5:
            #filename = "%s.png" % name
            #cv2.imwrite('test_output_50_200.png', frame)
            #break
            #out.write(frame)

    cv2.destroyAllWindows()
    out.release()

    return 0

path = '../data/data_for_visualization/wednesday_thursday_days_nights_rounded_only_ones.txt'

dataset = np.loadtxt(path)
#dataset = dataset[0:439724]

make_video(dataset)
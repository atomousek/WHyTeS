import numpy as np
import cv2
from datetime import datetime


# This function comes from https://stackoverflow.com/questions/44816682/drawing-grid-lines-across-the-image-uisng-opencv-python
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


def make_video_with_zeros (data):
    x_max = np.max(data[:, 1])
    x_min = np.min(data[:, 1])
    y_max = np.max(data[:, 2])
    y_min = np.min(data[:, 2])
    t_max = np.max(data[:, 0])
    t_min = np.min(data[:, 0])

    times = np.arange(t_min, t_max, 0.1, float)
    for i in range(len(times)):
        times[i] = round(times[i], 1)
    times = np.unique(times)

    width = int((x_max-x_min)*10)
    height = int((y_max-y_min)*10)
    number_of_frames = len(times)

    k = 0
    j = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('video.avi', fourcc, 20, (width, height))
    frame = cv2.cvtColor(np.full([height, width], 255, np.uint8), cv2.COLOR_GRAY2BGR)
    for i in xrange(number_of_frames):
        print str(i) + ' / ' + str(number_of_frames)
        if j%10 == 0:
            frame = cv2.cvtColor(np.full([height, width], 255, np.uint8), cv2.COLOR_GRAY2BGR)

        name = str(times[i])
        counter = 0
        #print 'i = ' + str(i) + '   k = ' + str(k) + '  times[i] = ' + str(times[i]) + '    data[k, 0] = ' + str(data[k, 0])
        while k < len(data[:, 0]) and times[i] == data[k, 0]:
            if data[k, 3] == 1:
                frame = cv2.circle(frame, (height - int(data[k, 2] + abs(y_min))*10, int(data[k, 1] + abs(x_min))*10), 2, (255, 0, 0), 1)
                counter += 1
                j = 0
            k += 1

        j += 1
        temp = frame
        temp = cv2.putText(temp, name, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), lineType=cv2.LINE_4)
        out.write(temp)

        #if counter > 5:
        #filename = "%s.png" % name
        #cv2.imwrite('test_output_50_200.png', frame)
        #break
        #out.write(frame)

    cv2.destroyAllWindows()
    out.release()

    return number_of_frames

def make_video (data):
    times = np.unique(data[:, 0])
    x_max = np.max(data[:, 1])
    x_min = np.min(data[:, 1])
    y_max = np.max(data[:, 2])
    y_min = np.min(data[:, 2])
    t_max = np.max(data[:, 0])
    t_min = np.min(data[:, 0])

    print 'x_min = ' + str(x_min) + '        x_max = ' + str(x_max) + '      y_min = ' + str(y_min) + '     y_max = ' + str(y_max)

    # Size of the frames will be ( x_range * size_factor ,  y_range * size_factor )
    size_factor = 10

    width = int((x_max-x_min)*size_factor)
    height = int((y_max-y_min)*size_factor)
    number_of_frames = len(times)

    k = 0
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter('video_with_only_ones.avi', fourcc, 20, (width, height))
    print str(width) + ',   ' + str(height)
    counter = 0
    empty_frame = cv2.cvtColor(np.full([height, width], 255, np.uint8), cv2.COLOR_GRAY2BGR)
    draw_grid(empty_frame, (150, 150, 150), 1, cv2.LINE_4, size_factor)
    door_counter = 0
    for i in xrange(number_of_frames):
       #print str(i) + ' / ' + str(number_of_frames)
        frame = empty_frame.copy()
        name = str(times[i])
        while k < len(data[:, 0]) and times[i] == data[k, 0]:
            if data[k, 3] == 1:
                #cv2.circle(frame, (height - int(data[k, 2] + abs(y_min))*10, int(data[k, 1] + abs(x_min))*10), 2, (255, 0, 0), 1)
                cv2.circle(frame, (int(data[k, 1] + abs(x_min))*size_factor, height - int(data[k, 2] + abs(y_min))*size_factor), int(0.2*size_factor), (255, 0, 0), int(0.1*size_factor))

                # Labeling THE door
                if (abs(data[k, 1] - 19.3) < 0.2 and abs(data[k, 2] + 2.3) < 0.2) or (abs(data[k, 1] - 1.31) < 0.1 and abs(data[k, 2] - 5.63) < 0.1):
                    door_counter += 1
                    cv2.circle(frame, (int(data[k, 1] + abs(x_min)) * size_factor, height - int(data[k, 2] + abs(y_min)) * size_factor),
                               int(0.2 * size_factor), (0, 0, 255), int(0.1 * size_factor))



            k += 1
        cv2.putText(frame, name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), lineType=cv2.LINE_4)
        cv2.putText(frame, datetime.utcfromtimestamp(times[i]).strftime('Day:%d, Time:%H:%M:%S'), (9*size_factor, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.06*size_factor, (0, 0, 255), lineType=cv2.LINE_4)
        counter += 1
        out.write(frame)

        #if counter > 5:
        #filename = "%s.png" % name
        #cv2.imwrite('test_output_50_200.png', frame)
        #break
        #out.write(frame)

    cv2.destroyAllWindows()
    out.release()

    print 'door counter = ' + str(door_counter)
    return counter

path = '../data/data_for_visualization/wednesday_thursday_days_nights_only_ones.txt'

dataset = np.loadtxt(path)
print len(dataset[:, 0])
#dataset = dataset[0:439724]

print make_video(dataset)
#make_video(dataset)
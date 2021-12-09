from imutils.video import FPS
import imutils
import cv2

#new detection every 15 frames
DETECTION_FPS = 12
#extract the haar features for cars
car_cascade = cv2.CascadeClassifier('cascade2.xml')

def intersect(rect1, rect2):
    x1,y1,w1,h1 = rect1
    x2,y2,w2,h2 = rect2
    # If one rectangle is on left side of other
    if (x1 > (x2+ w2) or x2 > (x1+w1)):
        return False
    # If one rectangle is above other
    if (y1 > (y2+h2) or y2 > (y1+h1)):
        return False
    return True

def filter_trackers(tracker_array, success_boxes):
    filtered_trackers = []
    filtered_trackers_box = []
    for _ in range(len(tracker_array)):
        i = len(trackers_array) - (_ + 1)
        approvable = True
        success, box = success_boxes[i]
        x,y,w,h = box
        if w > 50 or h > 70:
            approvable = False
        if not success:
            continue
        for j in range(len(filtered_trackers)):
            if intersect(filtered_trackers_box[j], box):
                approvable = False
                break
        if approvable:
            filtered_trackers.append(tracker_array[i])
            filtered_trackers_box.append(box)
    return filtered_trackers

def detect_vehicle(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_cascade.detectMultiScale(gray, 1.3, 1, minSize=(30,30), maxSize=(60,60))
    return cars


(major, minor) = cv2.__version__.split(".")[:2]

#choose the tracker algorithm
OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "boosting": cv2.TrackerBoosting_create,
    "mil": cv2.TrackerMIL_create,
    "tld": cv2.TrackerTLD_create,
    "medianflow": cv2.TrackerMedianFlow_create,
    "mosse": cv2.TrackerMOSSE_create
}
#default tracker is median flow base on our paper
tracker_mode = "medianflow"
BBtracker = OPENCV_OBJECT_TRACKERS[tracker_mode]()
#hold one tracker for each car
trackers_array = []
#video input initialization
vs = cv2.VideoCapture("video.avi")
# initialize the FPS throughput estimator
fps = FPS().start()


#first detection
ret, frame = vs.read()
frame = imutils.resize(frame, width=500)
cars = detect_vehicle(frame=frame)
for car in cars:
    car_tracker = OPENCV_OBJECT_TRACKERS[tracker_mode]()
    car_tuple = (car[0], car[1], car[2], car[3])
    car_tracker.init(frame, car_tuple)
    trackers_array.append(car_tracker)

# loop over frames from the video stream
frame_count = 0
while True:
    full_frame = vs.read()
    ret, frame = full_frame
    frame_count = frame_count + 1
    if frame is None:
        break
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    #detection again
    if ((frame_count % DETECTION_FPS) == 0) or (len(trackers_array) == 0):
        cars = detect_vehicle(frame=frame)
        for car in cars:
            car_tracker = OPENCV_OBJECT_TRACKERS[tracker_mode]()
            car_tuple = (car[0], car[1], car[2], car[3])
            car_tracker.init(frame, car_tuple)
            trackers_array.append(car_tracker)
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
    success_boxes_array = []
    for tracker in trackers_array:
        (success, box) = tracker.update(frame)
        success_boxes_array.append((success,box))
        if not success:
            trackers_array.remove(tracker)
        else:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

        # update the FPS counter
        fps.update()
        fps.stop()
        #print information
        info = [
            ("exit key ", "q"),
            ("Tracker ", "medianflow"),
            ("Success ", "Yes" if success else "No"),
            ("FPS ", "{:.2f}".format(fps.fps())),
        ]

        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                        cv2.FONT_ITALIC, 0.6, (0, 0, 0), 2)
    trackers_array = filter_trackers(trackers_array, success_boxes_array)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q") or key == ord("Q"):
        break

#release the vs
vs.release()
# close all windows
cv2.destroyAllWindows()
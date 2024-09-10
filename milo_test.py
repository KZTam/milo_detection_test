# from roboflow import Roboflow
# rf = Roboflow(api_key="IYNUlO2AxP9anNHN9zzV")
# project = rf.workspace("utm-gfonq").project("milo_test")
# version = project.version(1)
# dataset = version.download("yolov8")

# from ultralytics import YOLO
# model = YOLO('yolov8n.pt')
# model.train(data="C://Users/tamka/OneDrive/Documents/milo_detection_test/milo_test-1/data.yaml",epochs=50)

#########################################################

from ultralytics import YOLO
import cv2 as cv
import time
import math
# import numpy as np

confidence_threshold = 0.08
#confidence_threshold = 0.5 #for yolov8n.pt

milo_sum = 0
color_red = (0,0,255)
omega = 2*math.pi*0.2

model = YOLO('C://Users/tamka/OneDrive/Documents/milo_detection_test/runs/detect/train7/weights/best.pt')
#model = YOLO('C://Users/tamka/OneDrive/Documents/milo_detection_test/yolov8n.pt')
capture = cv.VideoCapture(0)
prev_time = time.time()

while capture.isOpened():

    rainbow_bgr = (255*math.sin(omega*prev_time),255*math.sin(omega*prev_time+(2/3)*math.pi),255*math.sin(omega*prev_time-(2/3)*math.pi))

    ret, frame = capture.read()
    frame = cv.flip(frame,1)

    if not ret:
        print("Unable to read data from CAM\n")
        break

    results = model(frame, conf=confidence_threshold)
    for result in results:
        for box in result.boxes:
            milo_sum = milo_sum + 1

            cls = box.cls[0].int().numpy()
            x1, y1, x2, y2 = box.xyxy[0].int().numpy()
            conf_level = box.conf.item()
            x_mid = abs(x1 - x2)/2 + x1
            y_mid = abs(y1 - y2)/2 + y1
            detect_str = "Class: %d start:(%d,%d) end:(%d,%d) pos:(%d,%d)\n" % (cls, x1, y1, x2, y2, x_mid, y_mid)
            print(detect_str)

            milo_label = "milo, %.2f" % (conf_level)
            cv.rectangle(frame,(x1,y1),(x2,y2),rainbow_bgr,2)
            cv.putText(frame,milo_label,(x1, y1 - 10),cv.FONT_HERSHEY_SIMPLEX,1,rainbow_bgr,2)

    #image_processed = results[0].plot()    #plot boxes using yolov8 library, 'image_processed' is also Mat vartype

    #fps = capture.get(cv.CAP_PROP_FPS)  
    #frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    fps = 1/(time.time() - prev_time)
    fps_str = "fps: %.2f, milo: %d" % (fps,milo_sum)
    cv.putText(frame,fps_str,(40,40),cv.FONT_HERSHEY_SIMPLEX,1,rainbow_bgr,2)
    cv.imshow("MILO DETECTION",frame)
    prev_time = time.time()
    milo_sum = 0

    if (cv.waitKey(1) == ord('q')):
        break;

cv.destroyAllWindows()
capture.release()
import cv2
import os
import cvlib as cv
import numpy as np
from demo import Predict
import serial
import time
webcam = cv2.VideoCapture("test_kimkhoa.mp4")
arduino = serial.Serial(port='COM11', baudrate=115200, timeout=.1)
#webcam = cv2.VideoCapture(0)
# loop through frames
# def write_read(x):
#     arduino.write(b(x, 'utf-8'))
#     time.sleep(0.05)
#     data = arduino.readline()
#     return data
i=0
while True:
    # read frame from webcam 
    status, frame = webcam.read()
    #frame=cv2.resize(frame,(640,640))
    # apply face detection
    if i%3==0:
        
        face, confidence = cv.detect_face(frame)
        print(confidence)
        # loop through detected faces
        for idx, f in enumerate(face):
            # get corner points of face rectangle        
            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]
            # draw rectangle over face
            
            # crop the detected face region
            face_crop = np.copy(frame[startY:endY,startX:endX])
            if (face_crop.shape[0]) < 10 or (face_crop.shape[1]) < 10:
                continue
            # preprocessing for gender detection model
            face_crop = cv2.resize(face_crop, (64,64))
            a=Predict(face_crop)
            Y = startY - 10 if startY - 10 > 10 else startY + 10
            # if a=='khoa':
            #     arduino.write(a.encode())
            #     print(arduino.readline().decode('ascii'))
            #     time.sleep(0.5)
        
        
            if a=='khoa':
                cv2.rectangle(frame, (startX,startY), (endX,endY), (0,255,0), 2)
                cv2.putText(frame, a, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)
                # arduino.write(a.encode())
                # print(arduino.readline().decode('ascii'))
                time.sleep(0.5)
            
            
            #cv2.putText(frame, a, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
            #           0.7, (0, 255, 0), 2)
            # if a !='nien':
            
            cv2.imshow("crop", face_crop)
            cv2.imshow("gender detection", frame)
            cv2.imwrite("result.jpg",frame )
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    i+=1
webcam.release()
cv2.destroyAllWindows()

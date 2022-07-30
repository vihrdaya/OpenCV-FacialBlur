import numpy as np
import cv2 as cv
from scipy.ndimage import gaussian_filter
#from pylab import *

video = cv.VideoCapture(0)

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')


while True:

    ret, frame = video.read()
    faces = face_cascade.detectMultiScale(frame, 1.3, 5)
    
    for (x, y, w, h) in faces:
#       cv.rectangle(frame,(x,y),(x+w, y+h),(255, 0, 0), 2)
        roi_color = frame[y:y+h, x:x+w]
        box = frame[y:y+h, x:x+w]
        croppedim = np.zeros(roi_color.shape)
        for i in range(3):
            croppedim[:, :, i] = gaussian_filter(roi_color[:, :, i], 20)

    frame[y:y+h, x:x+w] = croppedim

    cv.imshow('output', frame)
    k = cv.waitKey(1)
    if k == ord('q'):
        break
video.release()
cv.destroyAllWindows()


#cv.imshow("output", newim)

import numpy as np
import cv2
from scipy.ndimage import filters
from pylab import *

cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
img = cv2.imread("hyuk-hyuk.jpg")                        # Read image

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, 1.3, 5)

for (x,y,w,h) in faces:
#    img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    box = img[y:y+h, x:x+w]
    croppedim = np.zeros(roi_color.shape)
    for i in range(3):
        croppedim[:, :, i] = filters.gaussian_filter(roi_color[:, :, i], 20)

    croppedim = uint8(croppedim)

    newim = img
    newim[y:y+h, x:x+w] = croppedim




cv2.imshow("output", newim)                            # Show image
cv2.waitKey(0)           
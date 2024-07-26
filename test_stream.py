import sys
import cv2

vcap = cv2.VideoCapture("rtp://10.0.0.138:1234")

while(1):
	ret, frame = vcap.read()
	cv2.imshow("VIDEO", frame)
	cv2.waitKey(1)

import cv2
import numpy as numpy

cap = cv2.VideoCapture(0)

while True:
	_, frame = cap.read()
	hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

	lower_fire = numpy.array([150, 150,0])
	higher_fire = numpy.array([180, 255, 255])

	#canny = cv2.Canny(frame, 30, 150)
    #cv2.imshow("Canny", canny)
    #cv2.waitKey(0)

	mask = cv2.inRange( hsv, lower_fire, higher_fire)
	res = cv2.bitwise_and(frame, frame, mask = mask)

	cv2.imshow('frame', frame)
	#cv2.imshow("Canny", canny)
	cv2.imshow('res', res)
	k = cv2.waitKey(5) & 0XFF

	if k == 27:
		break
cv2.destroyAllWindows()
cap.release()
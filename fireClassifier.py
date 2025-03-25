import numpy as np
import cv2


#this is the cascade we just made. Call what you want
fire_cascade = cv2.CascadeClassifier('fire-cascade-11.xml')
peggy = fire_cascade.load('fire-cascade-11.xml')
print (peggy)

print("Indiwrw")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    # add this
    # image, reject levels level weights.
    fires = fire_cascade.detectMultiScale(gray, 50, 50)
    
    # add this
    for (x,y,w,h) in fires:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,'FIRE',(x-w,y-h) , font, 0.5, (0,255,255),2)


        #cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)

    
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib import colors
import glob
from sklearn.neighbors import KNeighborsClassifier
import mahotas

rootdir = "data"
X = []
Y = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:

        y = 1.0*(not "not_fire" in subdir)

        print(y,subdir)

        file_path = "{}/{}".format(subdir,file)
        print(file_path)
        img = cv2.imread(file_path)[:,:,::-1]


        #resizing image to 100 by 100 pixels
        img = cv2.resize(img,(100, 100))

        #adaptive color thresholding (Otsu) ---DOESN'T WORK WITH THIS SECTION, IDK WHY
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #found this online and used to try to catch the problem, assuming that the cv2.cvtColor is causing a problem
        #if img == None: 
            #raise Exception("could not load image !")

        #adaptive color thresholding cont'd
        blurred = cv2.GaussianBlur(img, (5,5), 0)
        cv2.imshow("Image", img)
        T = mahotas.thresholding.otsu(blurred)
        print ("Otsu's threshold: ", T)
        thresh = img.copy()
        thresh[thresh > T] = 255
        thresh[thresh < 255] = 0
        thresh = cv2.bitwise_not(thresh)
        cv2.imshow("Otsu", thresh)


        #Canny edge Detection -
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canny = cv2.Canny(img, 30, 150)
        cv2.imshow("Canny", canny)
        cv2.waitKey(0)

    #converting the image to a numerical array
        p = np.array(img)

    #flattening the numerical array to one vector
        p = p.flatten() #.reshape((1,-1))
        y = np.array(y)

        #making and array of flattened image arrays
        X.append(p)
        Y.append(y)
        #need to write this to a CSV file? in a loop to create comma separated list?

X = np.array(X)
Y = np.array(Y)
print(X.shape)
print(Y.shape)
plt.show(X)

#need to read in CSV

#data fitting model
#n_neighbors=1 means to choose the data point closest to the data point
t0 = time.time()
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X,Y)



#to make pedictions for new pobservations
cap = cv2.VideoCapture(0)
_, frame = cap.read()
frame = cv2.resize(frame,(100, 100))
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
f = np.array(frame)
f.flatten()
X_new =  f.reshape((1,-1)) #image to be tested
print(knn.predict(X_new))
print("total time: ", time.time() - t0)

  


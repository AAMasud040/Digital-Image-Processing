import cv2 as cv
import numpy as np
img = cv.imread('j.jpg',0)

kernel = np.ones((5,5),np.uint8)

dilation = cv.dilate(img,kernel,iterations = 1)
img_neg = 255 - dilation

gradient = cv.morphologyEx(img_neg, cv.MORPH_GRADIENT, kernel)

for i in range(gradient.shape[0]):
    for j in range(gradient.shape[1]):
        if(gradient[i][j]<45):
            gradient[i][j] = 0
        else:
            gradient[i][j] = 255

cv.imshow('gradient',gradient)

kernel = np.ones((5,5),np.uint8)
opening = cv.morphologyEx(gradient, cv.MORPH_OPEN, kernel)

dilation = cv.dilate(opening,kernel,iterations = 1)

kernel = np.ones((5,5),np.float32)/25
dst = cv.filter2D(opening,-1,kernel)

final = 255-dst
kernel = np.ones((5,5),np.uint8)
erosion = cv.erode(dst,kernel,iterations = 1)

final = 255-erosion
cv.imwrite('horseanimation.jpg', final)
cv.imshow('window_name', final)
  
#waits for user to press any key 
#(this is necessary to avoid Python kernel form crashing)
cv.waitKey(0) 
  
#closing all open windows 
cv.destroyAllWindows() 
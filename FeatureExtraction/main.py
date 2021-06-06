import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('K1002.jpg',0)
cv.imshow('image',img)
cv.waitKey(0)
ret,thresh = cv.threshold(img,127,255,0)
contours,hierarchy = cv.findContours(thresh, 1, 2)
cnt = contours[0]
# cv.drawContours(img,contours,-1,(128,255,0),3)
# cv.imshow("Contour", img)
# cv.waitKey(0)
#Moments
M = cv.moments(cnt)
print(M)
#Area
area = cv.contourArea(cnt)
print("Area: ", area)
#Perimeter
perimeter = cv.arcLength(cnt,True)
print("Perimeter: ",perimeter)
#is convex
k = cv.isContourConvex(cnt)
print("Is convex: ",k)

#Rectangle (minimum area)
x,y,w,h = cv.boundingRect(cnt)
cv.rectangle(img,(x,y),(x+w,y+h),(128,255,0),2)
rect = cv.minAreaRect(cnt)
box = cv.boxPoints(rect)
box = np.int0(box)
# cv.drawContours(img,[box],0,(128,255,0),2)
# cv.imshow("rectangle", img)
# cv.waitKey(0)
print("rect: ",rect)

#convex hull
hull = cv.convexHull(cnt)
#print("convex hull: ", hull)
#PolyDP
epsilon = 0.1*cv.arcLength(cnt,True)
approx = cv.approxPolyDP(cnt,epsilon,True)
print("Corners: ",len(approx))
# cv.drawContours(img, approx, -1, (128,255,0), 2)
# cv.imshow("approx", img)
# cv.waitKey(0)

#harris corners
gray = np.float32(img)
dst = cv.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv.dilate(dst,None)
#print(img[dst>0.01*dst.max()])
# Threshold for an optimal value, it may vary depending on the image.
dst2 = dst>0.01*dst.max()
print("Harris Corners: ",np.sum(dst2))
img[dst2] = 128
#img[dst>0.01*dst.max()]=[0,255,0]
cv.imshow('dst',img)
if cv.waitKey(0) & 0xff == 27:
    cv.destroyAllWindows()


# Shi-Tomasi Corner Detector
gray = np.float32(img)
corners = cv.goodFeaturesToTrack(gray,5,0.01,10)
corners = np.int0(corners)
print("Shi-Tomasi Corners: ",corners[:,0,0])
for i in corners:
    x,y = i.ravel()
    cv.circle(img,(x,y),0,255,0)
plt.imshow(img),plt.show()
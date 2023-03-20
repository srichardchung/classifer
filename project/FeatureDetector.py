import cv2 as cv
import numpy as np
import sys

img1 = cv.imread("ImagesQuery/Curious Incident.png", 0)
img2 = cv.imread("ImagesTrain\\curiousincident.JPG", 0)


orb = cv.ORB_create(nfeatures=1000)

kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

#print(des1)
#print(des2)
#sys.exit()

imgKp1 = cv.drawKeypoints(img1, kp1, None)
imgKp2 = cv.drawKeypoints(img2, kp2, None)


bf = cv.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)

print(len(good), "acceptable feature matches")

cv.imshow('img3', img3)
cv.waitKey(0)



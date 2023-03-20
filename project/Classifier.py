import cv2 as cv
import numpy as np
import os


# Import images and create class list
path = "ImagesQuery"
images = []
classNames = []
print(os.getcwd())
if "project" not in os.getcwd():
    os.chdir("ftr-detect-classifier/project")
print(os.getcwd())
myList = os.listdir(path)
print('Total Classes Detected:', len(myList))

for cls in myList:
    imgCur = cv.imread(f'{path}/{cls}', 0)
    images.append(imgCur)
    classNames.append(os.path.splitext(cls)[0])

# Create features used for matching
orb = cv.ORB_create(nfeatures=1000)

# Find descriptors for features
def findDes(images):
    desList = []
    for image in images:
        kp, des = orb.detectAndCompute(image, None)
        desList.append(des)
    return desList

desList = findDes(images)


# Match descriptors and find best match
def findID(img, desList):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList) != 0:
        if max(matchList) > 15:
            finalVal = matchList.index(max(matchList))
    return finalVal

desList = findDes(images)




# Configure camera and show output result as overlay in camera
cap = cv.VideoCapture(0)

while True:
    success, img2 = cap.read()
    imgOriginal = img2.copy()
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    id = findID(img2, desList)
    if id != -1:
        cv.putText(imgOriginal, classNames[id], (50,50), cv.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    cv.imshow('img2', imgOriginal)
    cv.waitKey(1)






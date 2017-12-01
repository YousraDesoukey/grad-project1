import numpy as np
import cv2
from matplotlib import pyplot as plt

img1 = cv2.imread('c2.jpg',0)          # queryImage
img2 = cv2.imread('Capture.PNG',0) # trainImage

# Initiate SIFT detector

orb = cv2.ORB_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)
# create BFMatcher object
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.

img3=cv2.drawMatches(img1,kp1,img2,kp2,matches[:20],None, flags=2)
x= []
y = []
#print(kp1[0].pt[0])
#print(kp1[0].pt[1])
for n in range(0,500):
    x.append(kp1[n].pt[0])

for n in range(0,500):
    y.append(kp1[n].pt[1])



x.sort()
y.sort()
#(kp1.pt[0]).sort()
#print(kp1.pt[0])
for n in range(0, 500):
    print(x[n])


for n in range(0, 500):
    print(y[n])

#print(len(kp1))



plt.imshow(img3),plt.show()
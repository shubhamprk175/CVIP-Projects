#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import os

UBIT = 'spareek'
np.random.seed(sum([ord(c) for c in UBIT]))
print("Code tested on OpenCV version : 3.4.1")
print("Your OpenCV version: {}".format(cv2.__version__))

OUTPUT_DIR = "./outputs/Task1_output/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

#%%
def show_image(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), aspect='auto', interpolation='nearest')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def save_image(img, fname="test"):
    cv2.imwrite(OUTPUT_DIR+fname+".jpg", img)
    return True



img1 = cv2.imread('images/mountain1.jpg')
img2 = cv2.imread('images/mountain2.jpg')
gray_img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#%%
######################################################
####################  Task 1  ########################
######################################################
# Initiatialize SIFT detector
sift = cv2.xfeatures2d.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, desc1 = sift.detectAndCompute(gray_img1, None)
kp2, desc2 = sift.detectAndCompute(gray_img2, None)

kp_img1 = cv2.drawKeypoints(img1, kp1, None, color=(255, 0, 0))
kp_img2 = cv2.drawKeypoints(img2, kp2, None, color=(255, 0, 0))
show_image(kp_img1)
show_image(kp_img2)
save_image(kp_img1, fname="task1_sift1")
save_image(kp_img2, fname="task1_sift2")

######################################################
####################  Task 2  ########################
######################################################
# Initialize Brute Force Mathher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

# Conside only the matches with distance < 0.75
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)



# Draw all the matches including outliers
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2, matchColor = (60,30,150))
show_image(img3)
save_image(img3, fname="task1_matches_knn")


#%%
######################################################
####################  Task 3  ########################
######################################################
# Compute Homography Matrix

src_pts = np.array([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
dst_pts = np.array([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
print(M)

#%%
######################################################
####################  Task 4  ########################
######################################################
# Draw 10 random inliers

# Convert Mast to 1D list
matchesMask = mask.ravel().tolist()

x = list(range(len(good)))
random.shuffle(x)
random_good = [ good[i] for i in x if matchesMask[i] != 0][:10]

img4 = cv2.drawMatches(img1, kp1, img2, kp2, random_good, None, flags=2, matchColor = (60,30,150))
show_image(img4)
save_image(img4, fname="task1_matches")



#%%
######################################################
####################  Task 5  ########################
######################################################
# Warp first image to secong image


stitcher = cv2.createStitcher(False)
result = stitcher.stitch((img1, img2))
save_image(result[1], fname="task1_pano")
show_image(result[1])

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
OUTPUT_DIR = "./outputs/Task2_output/"
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

#%%
img1 = cv2.imread('images/tsucuba_left.png')
img2 = cv2.imread('images/tsucuba_right.png')
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

kp_img1 = cv2.drawKeypoints(img1, kp1, None, color=(190, 190, 10))
kp_img2 = cv2.drawKeypoints(img2, kp2, None, color=(190, 190, 10))
show_image(kp_img1)
show_image(kp_img2)
save_image(kp_img1, fname="task2_sift1")
save_image(kp_img2, fname="task2_sift2")

# Initialize Brute Force Mathher
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

# Conside only the matches with distance < 0.75
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)


# Draw all the matches including outliers
img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2, matchColor = (190, 190, 10))
show_image(img3)
save_image(img3, fname="task2_matches_knn")

#%%
######################################################
####################  Task 2  ########################
######################################################

src_pts = np.array([ kp1[m.queryIdx].pt for m in good ])
dst_pts = np.array([ kp2[m.trainIdx].pt for m in good ])

FM, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.FM_RANSAC)
print(FM)

#%%
######################################################
####################  Task 3  ########################
######################################################

def drawlines(img1,img2,lines,pts1,pts2, color_flag_arr):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)

    for r,pt1,pt2,c_flag in zip(lines,pts1,pts2, color_flag_arr):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        color = (0, 255, 0) if c_flag else (0, 0, 255)
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,(int(pt1[0]), int(pt1[1])),5, color,-1)
        img2 = cv2.circle(img2,(int(pt2[0]), int(pt2[1])),5, color,-1)
    return img1,img2

def comp_points(p1, p2):
    if p1[0] == p2[0] and p1[1] == p2[1]:
        print("Same")
        return True
    return False

#%%
pts1 = src_pts[mask.ravel()==1]
pts2 = dst_pts[mask.ravel()==1]

lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1,1,2), 2, FM)
lines1 = lines1.reshape(-1,3)
done = False
while not done:
    r_pts1 = []
    r_pts2 = []
    r_lines1 = []
    color_flag_arr = []

    rand_list = random.sample(range(0, len(pts1)), 10)
    for j in rand_list:
        r_pts1.append(pts1[j])
        r_pts2.append(pts2[j])
        r_lines1.append(lines1[j])
        color_flag_arr.append(True)

    img5,img6 = drawlines(gray_img1,gray_img2,r_lines1, r_pts1, r_pts2, color_flag_arr)



    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1,1,2), 1, FM)
    lines2 = lines2.reshape(-1,3)

    r2_pts1 = []
    r2_pts2 = []
    r2_lines2 = []
    color_flag_arr_2 = []


    rand_list2 = random.sample(range(0, len(pts1)), 10)
    for i, j in enumerate(rand_list2):
        r2_pts1.append(pts1[j])
        r2_pts2.append(pts2[j])
        r2_lines2.append(lines1[j])
        color_flag_arr_2.append(comp_points(r2_pts2[i], r_pts2[i]))



    img3,img4 = drawlines(gray_img2, gray_img1, r2_lines2, r2_pts2, r2_pts1, color_flag_arr_2)


    for f in color_flag_arr_2:
        if f:
            save_image(img5, 'task2_epi_left')
            save_image(img3, 'task2_epi_right')
            show_image(img5)
            show_image(img3)
            done = True

#%%
######################################################
####################  Task 4  ########################
######################################################
stereo = cv2.StereoBM_create(numDisparities=64, blockSize=15)
disparity = stereo.compute(gray_img1, gray_img2)
save_image(disparity, "task2_disparity")
plt.imshow(disparity)

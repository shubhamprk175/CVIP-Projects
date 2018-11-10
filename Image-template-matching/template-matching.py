
import cv2
import numpy as np
import os


template = cv2.imread('template_1.jpg', cv2.IMREAD_GRAYSCALE)
template_L = cv2.Laplacian(template, cv2.CV_8U)
w, h = template.shape[::-1]



images = [
    'pos_1.jpg', 'pos_2.jpg', 'pos_3.jpg', 'pos_4.jpg', 'pos_5.jpg',
    'pos_6.jpg', 'pos_7.jpg', 'pos_8.jpg', 'pos_9.jpg', 'pos_10.jpg',
    'pos_11.jpg', 'pos_12.jpg', 'pos_13.jpg', 'pos_14.jpg', 'pos_15.jpg',
    'neg_1.jpg', 'neg_2.jpg', 'neg_3.jpg', 'neg_4.jpg', 'neg_5.jpg',
    'neg_6.jpg', 'neg_7.jpg', 'neg_8.jpg', 'neg_9.jpg', 'neg_10.jpg'
]


if not os.path.exists('./TemplateMatching'):
    os.makedirs('./TemplateMatching/')
for image in images:
    img = cv2.imread('images/task3/' + image, 0)
    img_G = cv2.GaussianBlur(img, (5, 5), 0)
    img_G_L = cv2.Laplacian(img_G, cv2.CV_8U)

    result = cv2.matchTemplate(img_G_L, template_L, cv2.TM_CCOEFF)
    print(result)
    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(result)
    top_left_corner = maxLoc
    bottom_right_corner = (top_left_corner[0] + w, top_left_corner[1] + h)

    cv2.rectangle(img, top_left_corner, bottom_right_corner, 255, 2)

    cv2.imwrite('./TemplateMatching/' + image, img)



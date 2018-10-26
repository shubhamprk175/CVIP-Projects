import cv2
import numpy as np
import os.path
import math



def convolve(img, kernel):
    height, width = img.shape
    # Because img is the padded image and we need the shape of unpadded_img
    height -= 2
    width -= 2

    result_img = np.asarray([[0.0 for _ in range(width+2)] for _ in range(height+2)])

    for i in range(1, height + 1):
        for j in range(1, width + 1):
            # Extracting the 3X3 array which needs to be multiplied with 3X3 kernel
            square_3X3 = img[i - 1:i + 2, j - 1:j + 2]
            members = square_3X3 * kernel
            total = sum([sum(item) for item in members])
            result_img.itemset((i - 1, j - 1), total)
    return result_img

def normalize(gradient):
    gradient_min = min([min(x) for x in gradient])
    gradient_max = max([max(x) for x in gradient])

    return (gradient - gradient_min) / (gradient_max - gradient_min)
    

unpadded_img = cv2.imread("images/task1.png", cv2.IMREAD_GRAYSCALE)
height, width = unpadded_img.shape


img = np.asarray([[0.0 for _ in range(width+2)] for _ in range(height+2)])
img[1:height + 1, 1:width + 1] = unpadded_img

# Sobel Kernels for both X & Y Gradient
x_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
y_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

# X-Direction Edge
Gx = convolve(img, x_kernel)

# X-Direction Normalize
Gx = normalize(Gx)
cv2.imshow('Normalized_Gx_dir', Gx)

# Y-Direction Edge
Gy = convolve(img, y_kernel)


# Y-Direction Normalize
Gy = normalize(Gy)
cv2.imshow('Normalized_Gy_dir', Gy)

Resultant = sqrt(Gx^2 + Gy^2)
magnitude_img = math.sqrt(Gx ** 2 + Gy ** 2)
cv2.imshow("Magnitude Image", magnitude_img)


OUTPUT_DIR = "./outputs/EdgeDetection/"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# To bring image in integer range 0-255
Gx *= 255
Gy *= 255
magnitude_img *= 255


cv2.imwrite(OUTPUT_DIR+"Sobel_X.jpg", Gx)
cv2.imwrite(OUTPUT_DIR+"Sobel_Y.jpg", Gy)
cv2.imwrite(OUTPUT_DIR+"Magnitude.jpg", magnitude_img)

cv2.waitKey(0)
cv2.destroyAllWindows()

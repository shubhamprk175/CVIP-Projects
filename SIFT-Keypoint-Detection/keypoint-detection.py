import cv2
import math
import numpy as np
import os.path

class ut:

    def __init__(self):
        ''' Constructor for this class. '''
        # Create some member animals

    @staticmethod
    def f(x, y, sigma):
        """
        Gausssian Function
        :param x:
        :param y:
        :param sigma:
        :return: float, Calculated value of Gaussian Function
        """
        return (1/(2*math.pi*(sigma**2))) * math.exp(-(x**2+y**2)/(2*(sigma**2)))

    @staticmethod
    def create_kernel(sigma):
        """
        :param sigma: variance
        :return: 2D list, 7x7 Gaussion kernel
        """
        # sigma = 1/math.sqrt(2)
        return [[ut.f(x, y, sigma) for x in range(-3, 4)] for y in range(3, -4, -1)]

    @staticmethod
    def convolve(kernel, img):
        """
        Convolve image with gaussian kernel
        :param kernel: Gaussian kernel for each image in one octave
        :param img: image
        :return: np.array, Blurred Image
        """
        # Matrix Transpose code
        kernel = [[kernel[j][i] for j in range(len(kernel))] for i in range(len(kernel[0]))]
        height, width = img.shape

        # Pixel location of the image in padded image
        offset = len(kernel)//2  # 3 for 7x7

        # Because img is the padded image and we need the shape of unpadded_img
        height -= 6
        width -= 6

        # Blank image with zeros
        # result_img = np.zeros((height, width), np.float32)
        result_img = np.asarray([[0.0 for _ in range(width)] for _ in range(height)])

        for i in range(offset, height + offset):
            for j in range(offset, width + offset):
                # Extract 7x7 matrix and multiply it and add all the values
                square_7x7 = img[i-offset:i+offset+1, j-offset:j+offset+1]
                members = square_7x7 * kernel
                total = sum([sum(item) for item in members])

                result_img.itemset((i-3, j-3), total)
        return result_img

    @staticmethod
    def extract_neighbours(i, j, top, middle, bottom):
        """
        :param i: Row of the point in Middle DoG
        :param j: Column of the point in Middle DoG
        :param top: Upper DoG
        :param middle: Middle DoG
        :param bottom: Bottom DoG
        :return: List, containing all the values of 3x3 patch of each DoG
        """
        cube_3x3x3 = []
        cube_3x3x3.extend([ y for x in top[i-1:i+2, j-1:j+2] for y in x])
        cube_3x3x3.extend([ y for x in middle[i-1:i+2, j-1:j+2] for y in x])
        cube_3x3x3.extend([ y for x in bottom[i-1:i+2, j-1:j+2] for y in x])

        return cube_3x3x3

    @staticmethod
    def is_extrema(cube_3x3x3):
        """

        :param cube_3x3x3: List containing all the values of 3x3 patch of each DoG
        :return: Boolean, If the point is a minima or maxima
        """
        l = len(cube_3x3x3)
        central_pixel = cube_3x3x3[l//2]
        cube_3x3x3.pop(l//2)

        if central_pixel > max(cube_3x3x3) or central_pixel < min(cube_3x3x3):
            return True
        return False


    @staticmethod
    def kernel_normalize(kernel):
        """
        :param kernel: 7x7 Gaussian kernel
        :return: np.array, Gaussian kernel divided by sum of all the values
        """
        # Divide kernel values with sum of the kernel matrix
        return np.asarray(kernel, dtype=np.float32) / sum([sum(row) for row in kernel])

    @staticmethod
    def pad_image(image):
        """
        :param image: Image to padded
        :return: np.array, padded image
        """
        h, w = image.shape
        temp_img = np.asarray([[0 for _ in range(w+6)] for _ in range(h+6)])
        temp_img[3:h+3, 3:w+3] = image
        return temp_img


    @staticmethod
    def read_image(path):
        """
        :param path: path of the image to be read
        :return: np.array, Image read
        """
        temp = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return temp

    @staticmethod
    def scale_img(img):
        """
        Downsize the image by half
        :param img: Image to be resized
        :return: List, of all the resized images
        """
        oct_img = [img]
        for _ in range(3):
            oct_img.append(oct_img[-1][::2, ::2])
        return oct_img


def create_blur_matrix():
    """
    :return: Matrix having sigma values for all octaves
    """
    blur_matrix = []
    SIG = math.sqrt(2)
    sigma_vals = [1/math.sqrt(2), math.sqrt(2), 2*math.sqrt(2), 4*math.sqrt(2)]

    for sVal in sigma_vals:
        blur_octave = [sVal]
        for j in range(4):
            sVal *= SIG
            blur_octave.append(sVal)
        blur_matrix.append(blur_octave)
    return np.asarray(blur_matrix)


def create_scale_space(image):
    scale_space = []
    scaled_images = ut.scale_img(image)
    for blur_octave, sImg in zip(blur_matrix, scaled_images):
        octave = []
        sImg = ut.pad_image(sImg)
        for sVal in blur_octave:
            octave.append(ut.convolve(ut.kernel_normalize(ut.create_kernel(sVal)), sImg))
        scale_space.append(octave)
    return scale_space


def create_DoG_tree():
    DoG_tree = []
    for octave in scale_space:
        DoG_tree.append([y - x for x, y in zip(octave, octave[1:])])
    return DoG_tree


def save_scale_space_images():
    OUTPUT_DIR = "./outputs/scale_space/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for o, octave in enumerate(scale_space):
        for i, image in enumerate(octave):
            # name = "Octave_" + str(o + 1) + "_Image_" + str(i + 1) + ".png"
            cv2.imwrite(OUTPUT_DIR+"Octave_"+str(o)+"_"+str(i)+".png", np.asarray(image, np.uint8))


def save_dog_images():
    OUTPUT_DIR = "./outputs/DoG/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for o, octave in enumerate(DoG_tree):
        for i, image in enumerate(octave):
            # name = "Octave_" + str(o + 1) + "_Image_" + str(i + 1) + ".png"
            cv2.imwrite(OUTPUT_DIR+"Octave_"+str(o)+"_"+str(i)+".png", np.asarray(image, np.uint8))


def save_key_point_images():
    OUTPUT_DIR = "./outputs/key_points/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for o, octave in enumerate(keypoint_image_tree):
        for i, image in enumerate(octave):
            # name = "Octave_" + str(o + 1) + "_Image_" + str(i + 1) + ".png"
            cv2.imwrite(OUTPUT_DIR+"Octave_"+str(o)+"_"+str(i)+".png", np.asarray(image, np.uint8))


def save_project_key_point_images():
    OUTPUT_DIR = "./outputs/project_key_points/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    for o, image in enumerate(keypoint_projected_img):
        cv2.imwrite(OUTPUT_DIR+"Octave_"+str(o)+".png", np.asarray(image, np.uint8))


def detect_keypoints():
    keypoint_image_tree = []
    for o, octave in enumerate(DoG_tree):
        keypoint_octave = []
        for d, DoG in enumerate(octave):
            if d in [1, 2]:
                h, w = DoG.shape
                keypoint_img = np.asarray([[0 for _ in range(w)] for _ in range(h)], np.uint8)
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        cube_3x3x3 = ut.extract_neighbours(i, j, octave[d-1], DoG, octave[d+1])
                        if ut.is_extrema(cube_3x3x3):
                            keypoint_img.itemset((i, j), 255)
                keypoint_octave.append(keypoint_img)

        keypoint_image_tree.append(keypoint_octave)
    return keypoint_image_tree


def project_keypoints(image):
    """
    :param image: Image, find 4 scale of it, and locate the keypoint it
    :return: Image with keypoints on it, and Five left most keypoints from top-left corner
    """
    scaled_images = ut.scale_img(image)
    keypoint_projected_img = []
    for k_oct, img in zip(keypoint_image_tree, scaled_images):
        k1 = k_oct[0]
        k2 = k_oct[1]

        for i in range(len(k1)):

            for j in range(len(k1[0])):
                if k1[i][j] == 255 or k2[i][j] == 255:
                    img[i][j] = 255

        keypoint_projected_img.append(img)
    return keypoint_projected_img

def five_left_most_keypoints():
    kp = keypoint_image_tree[0][0] + keypoint_image_tree[0][1]
    keypoints = []
    count = 0
    for r, row in enumerate(kp):
        for c, col in enumerate(row):
            if count%5 != 0 and col >= 255:
                keypoints.append((r,c))
                count+=1
            else:
                break
        if count == 25:
            break
    return keypoints






img = ut.read_image("images/task2.jpg")
img = ut.pad_image(img)

blur_matrix = create_blur_matrix()

scale_space = create_scale_space(img)
save_scale_space_images()

DoG_tree = create_DoG_tree()
save_dog_images()

keypoint_image_tree = detect_keypoints()
save_key_point_images()

keypoint_projected_img = project_keypoints(img)
save_project_key_point_images()

five_key_points = five_left_most_keypoints()

cv2.imshow("Octave_1_Image_1", img)
cv2.waitKey(0)

for kp in five_key_points:
    print(kp)

# for o, octave in enumerate(scale_space):
#     for i, img in enumerate(octave):
#         cv2.imshow("Octave_"+str(o+1)+"_Image_"+str(i+1), img)
# cv2.waitKey(0)
#
#
# for o, octave in enumerate(DoG_tree):
#     for i, img in enumerate(octave):
#         cv2.imshow("Octave_"+str(o+1)+"_DoG_"+str(i+1), img)
# cv2.waitKey(0)
#
# for o, octave in enumerate(keypoint_image_tree):
#     for i, img in enumerate(octave):
#         cv2.imshow("Octave_"+str(o+1)+"_KeyPoint_"+str(i+1), img)
# cv2.waitKey(0)



import numpy as np
import cv2
import pickle

# constants
mtx_path = 'calib_data_pickle.p'

# read the pickled calibration data
with open(mtx_path, "rb") as input_file:
    calib_data = pickle.load(input_file)
mtx = calib_data["mtx"]
dist = calib_data["dist"]


# undistort the given image
def undistort(img):
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    return undist


def yellow_color_finder(img):
    hue = 15 # for yellow color in HSV space

    # define upper and lower ranges for detecting yellow color
    lower_range = np.array([max(0, hue - 10), 0, 0], dtype=np.uint8)
    upper_range = np.array([min(180, hue + 10), 255, 255], dtype=np.uint8)

    # convert the image to HSV space
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # mask the image in this range
    mask = cv2.inRange(img_hsv, lower_range, upper_range)

    # get a binary image
    binary_img = cv2.bitwise_and(img_hsv, img_hsv, mask=mask)
    binary_img = cv2.cvtColor(binary_img, cv2.COLOR_RGB2GRAY)

    # threshold the binary image to get only the thresholded pixels
    _, binary_img = cv2.threshold(binary_img, 127, 255, cv2.THRESH_BINARY)
    return binary_img


# applies sobel in the vertical direction and thresholds it to find the edges
#  of lane line
def mag_thresh(img, sobel_kernel=9, min_thresh=50, max_thresh=255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    sobel_x = cv2.Sobel(gray,
                        cv2.CV_64F, 1, 0,
                        ksize=sobel_kernel)
    scaled_sobel = np.int8(255 * np.absolute(sobel_x) / np.max(np.absolute(sobel_x)))
    mag_binary = np.zeros_like(scaled_sobel)
    mag_binary[(scaled_sobel >= min_thresh) & (scaled_sobel <= max_thresh)] = 1
    return mag_binary


# detects white lines on the road
def white_color_finder(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    eq_global = cv2.equalizeHist(gray)

    _, th = cv2.threshold(eq_global, thresh=250, maxval=255,
                          type=cv2.THRESH_BINARY)

    return th


# combines all the techniques to detect lane lines and returns a combined
# thresholded binary image
def combined_binary_image(image):
    undist = undistort(image)
    yellow = yellow_color_finder(undist)
    white = white_color_finder(undist)
    sobel_img = mag_thresh(undist)
    combined = np.zeros_like(white)
    combined[(yellow != 0) | (white != 0) | (sobel_img != 0)] = 1
    return combined


# returns the prespective or birds-eye view of the given image
def get_perspective(image):
    height, width = image.shape[:2]

    # source coordinates
    src_coords = np.float32([[width, height - 10],
                             [0, height - 10],
                             [546, 460],
                             [732, 460]])
    # destination coordinates
    dst_coords = np.float32([[width, height],
                             [0, height],
                             [0, 0],
                             [width, 0]])

    # Perspective Transform matrix
    M = cv2.getPerspectiveTransform(src_coords, dst_coords)

    img_size = (image.shape[1], image.shape[0])
    perspective_img = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    Minv = cv2.getPerspectiveTransform(dst_coords, src_coords)
    return perspective_img, Minv
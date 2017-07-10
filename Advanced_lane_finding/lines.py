# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2

# a line can be left or right
# every line has a detected flag which is updated every time the line is
# detected
# last fit is updated in case we do not find some lane and we fill it with the
# previous fit
# recent_fit is the current fit of the lane line and is stored every time a
# line is detected
# all the x and y pixels of the lane are also stored


class Line:
    """
    Class to model a lane-line.
    """
    def __init__(self, buffer_len=10):

        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit = None

        # polynomial coefficients fitted on the current iteration
        self.recent_fit = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None


# takes the left and right lane instances and returns the updated lines
def find_new_line(left_lane, right_lane, warp_image):

    # find histogram on half the image
    histogram = np.sum(warp_image[warp_image.shape[0] / 2:, :], axis=0)
    # get mid points of peaks
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((warp_image, warp_image, warp_image)) * 255
    # number of sliding windows
    nwindows = 9
    # setting height of windows
    window_height = np.int(warp_image.shape[0] / nwindows)
    # identify the x and y positions of all the non-zero pixels in the image
    nonzero = warp_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # width of window +/- margin
    margin = 100
    # minimum amount of pixels found to recenter window
    minpix = 50
    # create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # step through the windows one by one
    for window in range(nwindows):
        win_y_low = warp_image.shape[0] - (window + 1) * window_height
        win_y_high = warp_image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)
        # identify the nonzero pixels within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
        nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
        nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # if we find > 50 minpixels, recenter the next window on the mean of their position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # concatenate the arrays
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    left_lane.all_x = nonzerox[left_lane_inds]
    left_lane.all_y = nonzeroy[left_lane_inds]
    right_lane.all_x = nonzerox[right_lane_inds]
    right_lane.all_y = nonzeroy[right_lane_inds]

    left_lane.detected = True
    right_lane.detected = True

    # Fit a second order polynomial to each
    # if the lists are empty i.e. no lane pixels were found in this image
    # reach out to the last saved fits for rescue
    if not list(left_lane.all_x) or not list(left_lane.all_y):
        left_fit = left_lane.last_fit
        left_lane.detected = False
    else:
        left_fit = np.polyfit(left_lane.all_y, left_lane.all_x, 2)
        left_lane.last_fit = left_fit

    # same with right lane
    if not list(right_lane.all_x) or not list(right_lane.all_y):
        right_fit = right_lane.last_fit
        right_lane.detected = False
    else:
        right_fit = np.polyfit(right_lane.all_y, right_lane.all_x, 2)
        right_lane.last_fit = right_fit

    left_lane.recent_fit = left_fit
    right_lane.recent_fit = right_fit

    return left_lane, right_lane, out_img


# returns the calculated radius of curvature in real space and location of
# lines in pixel space for a given line
def get_roc_and_lane_pos(lane):
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    # y_eval is to get the location of the point where we want to find the
    # distance from center
    y_eval = np.max(ploty)

    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    lane_y = lane.all_y
    lane_x = lane.all_x

    # Fit new polynomials to x,y in world space
    fit_meters = np.polyfit(lane_y * ym_per_pix, lane_x * xm_per_pix, 2)
    # Calculate the new radii of curvature
    roc_meters = ((1 + (2 * fit_meters[0] * y_eval * ym_per_pix + fit_meters[1]) **
                   2) ** 1.5) / np.absolute(2 * fit_meters[0])

    # get the fit of pixels in pixels space
    fit = lane.recent_fit
    # position of the lane
    lane_pos = fit[0] * y_eval ** 2 + fit[1] * y_eval + fit[2]

    return roc_meters, lane_pos


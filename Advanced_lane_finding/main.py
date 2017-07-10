# this is the starting point of the lane finding process

import cv2
import numpy as np
import image_processing
import lines
import glob
from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt

# collect all test images
test_files_path = 'test_images/'
test_images = glob.glob(test_files_path + 'test*.jpg')
test_images.append(test_files_path + 'straight_lines1.jpg')
test_images.append(test_files_path + 'straight_lines2.jpg')

# instantiate lane lines
left_lane = lines.Line()
right_lane = lines.Line()


# a function that can plot two plots in one view
# useful for plotting in writeup
def plot_2pictures(image1, image1_title, image2, image2_title, cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(image1)
    ax1.set_title(image1_title, fontsize=30)
    ax2.imshow(image2, cmap=cmap)
    ax2.set_title(image2_title, fontsize=30)


# draws left and right lanes on the image
# along with radius of curvature and distance from center
def draw(left_lane, right_lane, img, Minv):
    ploty = np.linspace(0, 719, num=720)  # to cover same y-range as image
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # get the recent fit of the lane lines
    current_fit_left = left_lane.recent_fit
    current_fit_right = right_lane.recent_fit

    # Generate x and y values for plotting
    left_fitx = current_fit_left[0] * ploty ** 2 + \
                current_fit_left[1] * ploty + current_fit_left[2]
    right_fitx = current_fit_right[0] * ploty ** 2 + \
                 current_fit_right[1] * ploty + current_fit_right[2]

    # draw road as green polygon on original frame
    road_warp = np.zeros_like(img, dtype=np.uint8)
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    cv2.fillPoly(road_warp, np.int_([pts]), (0, 255, 0))
    road_dewarped = cv2.warpPerspective(road_warp, Minv, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, 1, road_dewarped, 0.3, 0)

    # calculate the mean radius of curvature for the given lines
    roc_left, left_pos = lines.get_roc_and_lane_pos(left_lane)
    roc_right, right_pos = lines.get_roc_and_lane_pos(right_lane)
    roc_mean = roc_left + roc_right/2.
    roc_mean = "{0:.2f}".format(roc_mean)
    curvature_string = 'Curvature: ' + roc_mean + 'm'
    cv2.putText(result, curvature_string, (800, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                255, 3)

    # calculate the distance from center of the car
    deviation = ((right_pos + left_pos) / 2. - 1280 / 2.) * xm_per_pix
    deviation = "{0:.2f}".format(deviation)
    deviation_string2 = 'Dist. from center: ' + deviation + 'm'
    cv2.putText(result, deviation_string2, (800, 200), cv2.FONT_HERSHEY_SIMPLEX,
                1, 255, 3)
    return result


# this is the pipeline which performs every step of the process of finding a
# lane line on the road
def pipeline(image):
    global left_lane, right_lane
    binary_image = image_processing.combined_binary_image(image)
    perspective_image, Minv = image_processing.get_perspective(binary_image)
    left_lane, right_lane, out_img = lines.find_new_line(left_lane,
                                                           right_lane,
                                                perspective_image)
    new_image = draw(left_lane, right_lane, image, Minv)
    return new_image

# img = plt.imread(test_images[0])
# ret_image = pipeline(img)
# ret_image = cv2.cvtColor(ret_image, cv2.COLOR_BGR2RGB)
# cv2.imwrite("output_images/result_test1.jpg", ret_image)


# call the pipeline on the project video and save it to a new video file
video_output = 'project_video_output.mp4'
clip1 = VideoFileClip("project_video.mp4")
video_clip = clip1.fl_image(pipeline)
video_clip.write_videofile(video_output, audio=False)



undistort_left_4 = cv2.imread("~/Farmtopia/Camera-Model-and-Stereo-Depth-Sensing/output/task_2/undistort_left_images_0.png")
undistort_right_4 = cv2.imread("~/Farmtopia/Camera-Model-and-Stereo-Depth-Sensing/output/task_2/undistort_right_images_1.png")
rectified_left_4 = cv2.imread("~/Farmtopia/Camera-Model-and-Stereo-Depth-Sensing/output/task_2/rectified_left_images_0.png")
rectified_right_4 = cv2.imread("~/Farmtopia/Camera-Model-and-Stereo-Depth-Sensing/output/task_2/rectified_right_images_1.png")

# Step 2: Block match for each pixel on the images to obtain a disparity map
matcher_left = cv2.StereoSGBM_create(minDisparity=0,
                                     numDisparities=160,
                                     blockSize=5,
                                     P1=8 * 3 * 3 ** 2,
                                     P2=32 * 3 * 3 ** 2,
                                     disp12MaxDiff=1,
                                     uniquenessRatio=15,
                                     speckleWindowSize=0,
                                     speckleRange=2,
                                     preFilterCap=63,
                                     mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
                                     )

matcher_right = cv2.ximgproc.createRightMatcher(matcher_left)
lamda = 1000
sigma = 1.2
v_multi = 0.5
filter_w = cv2.ximgproc.createDisparityWLSFilter(matcher_left=matcher_left)
filter_w.setLambda(lamda)
filter_w.setSigmaColor(sigma)
display_disparity = matcher_left.compute(rectified_left_4, rectified_right_4)  
display_rectified = matcher_right.compute(rectified_right_4, rectified_left_4)  
display_disparity = np.int16(display_disparity)
display_rectified = np.int16(display_rectified)
disparity = filter_w.filter(display_disparity, rectified_left_4, None, display_rectified) 
disparity = cv2.normalize(src=disparity, dst=disparity, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
disparity = np.uint8(disparity)
# Step 3: Calculate depth for each pixel using the disparity map
depth = cv2.reprojectImageTo3D(display_disparity, Q)
path = "../../output/task_4"
cv2.imwrite(os.path.join(path,"Rectified image left.png"), rectified_left_4)
cv2.imwrite(os.path.join(path,"Rectified image right.png"), rectified_right_4)
cv2.imwrite(os.path.join(path,"Disparity.png"), disparity)
# Step 4: Check the dense depth results
cv2.imshow("Rectified Image Left",rectified_left_4)
cv2.imshow("Rectified Image Right",rectified_right_4)
cv2.imshow("Disparity",disparity)
cv2.waitKey(0)

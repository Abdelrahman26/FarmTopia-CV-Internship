# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

#Convert from BGR to RGB
def cvt_BR(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

#Display two images
def display_2images(img1, img2):
    plt.figure(figsize=[20, 10]);
    plt.subplot(121);plt.axis('off');plt.imshow(img1);plt.title("Left Form")
    plt.subplot(122);plt.axis('off');plt.imshow(img2);plt.title("Right Form")

#Display one image
def display_image(img, title):
    plt.figure(figsize=[40, 10])
    plt.imshow(img)
    plt.axis('off')
    plt.title(title)

def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Read reference image
img_paths = glob.glob('Image_input/*.jpg')
imgL = cv2.imread(img_paths[1])
imgL = cvt_BR(imgL)
imgR = cv2.imread((img_paths[0]))
imgR = cvt_BR(imgR)


# Display Images
display_2images(imgL, imgR)


# Convert images to grayscale
imgL_gray = to_gray(imgL)
imgR_gray = to_gray(imgR)

# Detect ORB features and compute descriptors.
MAX_NUM_FEATURES = 150
orb = cv2.ORB_create(MAX_NUM_FEATURES)
keypoints1, descriptors1 = orb.detectAndCompute(imgL_gray, None)
keypoints2, descriptors2 = orb.detectAndCompute(imgR_gray, None)



imgL_display = cv2.drawKeypoints(imgL, keypoints1, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
imgR_display = cv2.drawKeypoints(imgR, keypoints2, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Display drwan key points
display_2images(imgL_display, imgR_display)

# Match features.
matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
# vector of tuples
matches = matcher.match(descriptors1, descriptors2, None)# <= max_number_of features(hyper)


# Sort matches by score
matches = sorted(matches, key= lambda x:x.distance, reverse=False)

# Remove not so good matches
numGoodMatches = int(len(matches) * 0.1)
matches = matches[:numGoodMatches]

# Draw top matches
img_matches = cv2.drawMatches(imgL, keypoints1, imgR, keypoints2, matches, None)

#Display matched_image
display_image(img_matches, "Matched Form")

# Extract location of good matches
points1 = np.zeros((len(matches), 2), dtype=np.float32)
points2 = np.zeros((len(matches), 2), dtype=np.float32)

for i, match in enumerate(matches):
    points1[i, :] = keypoints1[match.queryIdx].pt
    points2[i, :] = keypoints2[match.trainIdx].pt


# Find homography
h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

# Use homography to warp image --> (0.1*key_points)
height, width, channels = imgL.shape
result = cv2.warpPerspective(imgR, h, 	(imgR.shape[1]+imgL.shape[1], imgR.shape[0]))
result[0:imgL.shape[0], 0:imgL.shape[1]] = imgL

# Display results
display_image(result, "Stitched Form")
plt.show()


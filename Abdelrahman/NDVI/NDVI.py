# Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Converting from bgr to rgb
def cvt_BR(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Histogram for one channel
def histogram1d(NDVI_channel, b = 30):
    # channel flatten
    NDVI_flat = NDVI_channel.flatten()
    plt.hist(NDVI_flat, bins=b, color='red', alpha=1, histtype='step')
    plt.show()

# Read bgr and nir
img_bgr = cv2.imread('0028_rgb.tiff')
img_nir = cv2.imread('0028_nir.tiff')


# Converting to rgb
img_rgb = cvt_BR(img_bgr)
img_nir = cvt_BR(img_nir)


# Extract red channel
red_channel = img_bgr[:,:,0].astype('float64')

# Extract one channel from nir (all channels are the same)
img_nir = img_nir[:,:,0].astype('float64')

# Calculation of NDVI Channel
alpha = 0
numerator = red_channel - img_nir
denominator = (red_channel + img_nir) + alpha
NDVI_channel = numerator / denominator
#histogram1d(NDVI_channel)
NDVI_channel += 1
NDVI_channel /= 2
NDVI_channel *= 255

# Create Copy from img_rgb then add NDVI_channel in channel number 1
Colorized_NDVI = np.copy(img_rgb)
Colorized_NDVI[:,:,1] = NDVI_channel

# range of NDVI Channel has been converted from [-1:1] to [0:255]
#print(np.amin(NDVI_channel))
#print(np.amax(NDVI_channel))

# Trunc func
#print(NDVI_channel[19][16])
#print(Colorized_NDVI[19][16][1])


# Display rgb_image and colorized NDVI
plt.figure(figsize=[20, 10])
plt.subplot(121);plt.axis('off');plt.imshow(img_rgb)
plt.subplot(122);plt.axis('off');plt.imshow(Colorized_NDVI)
plt.show()

# save Colorized_NDVI img
cv2.imwrite('Colorized_NDVI.png', Colorized_NDVI)
#print(np.amin(Colorized_NDVI[:,:,2]))
#print(np.amax(Colorized_NDVI[:,:,2]))





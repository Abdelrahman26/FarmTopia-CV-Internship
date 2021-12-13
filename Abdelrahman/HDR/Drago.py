import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

def set_imgs():
    images = []
    filenames = glob.glob('/*.jpg')
    for filename in filenames:
        im = cv2.imread(filename)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        images.append(im)
    return images

# 4 esposurs
def set_times(t1, t2, t3, t4):
    # List of exposure times
    times = np.array([t1, t2, t3, t4], dtype=np.float32)
    return times

def readImagesAndTimes(images, times):
    return images, times

def imageAligned():
    # Read images and exposure times
    images, times = readImagesAndTimes()
    # Align Images
    # converts all the images to median threshold bitmaps (MTB).
    # An MTB for an image is calculated by assigning the value 1 to pixels brighter than median luminance and 0 otherwise.
    alignMTB = cv2.createAlignMTB()
    alignMTB.process(images, images)

def estimateCameraRespose():
    # Find Camera Response Function (CRF)
    calibrateDebevec = cv2.createCalibrateDebevec()
    # shape -> 256, 1, 3
    responseDebevec = calibrateDebevec.process(images, times)

    # Plot CRF
    x = np.arange(256, dtype=np.uint8)
    y = np.squeeze(responseDebevec)

    ax = plt.figure(figsize=(30, 10))
    plt.title("Debevec Inverse Camera Response Function", fontsize=24)
    plt.xlabel("Measured Pixel Value", fontsize=22)
    plt.ylabel("Calibrated Intensity", fontsize=22)
    plt.xlim([0, 260])
    plt.grid()
    plt.plot(x, y[:, 0], 'r', x, y[:, 1], 'g', x, y[:, 2], 'b');

def mergeExposure():
    mergeDebevec = cv2.createMergeDebevec()
    # images -> uint8
    # hdrDebevec -> float32
    hdrDebevec = mergeDebevec.process(images, times, responseDebevec)

def tonemapping():
    # Tonemap using Drago's method to obtain 24-bit color image (8-bit per channel) maping the 32-bit float HDR data into the range [0..1]
    # in some cases the values can be larger than 1 or lower the 0 (clib used)
    # gamma : This parameter compresses the dynamic range by applying a gamma correction. When gamma is equal to 1, no correction is applied. A gamma of less than 1 darkens the image, while a gamma greater than 1 brightens the image.
    # saturation : This parameter is used to increase or decrease the amount of saturation. When saturation is high, the colors are richer and more intense. Saturation value closer to zero, makes the colors fade away to grayscale.
    # contrast : Controls the contrast ( i.e. log (maxPixelValue/minPixelValue) ) of the output image.
    # bias : is the value for bias function in [0, 1] range. Values from 0.7 to 0.9 usually give the best results. The default value is 0.85.
    # The final output is multiplied by 3 just because it gave the most pleasing results.
    tonemapDrago = cv2.createTonemapDrago(gamma=1.2, saturation=1.0, bias=0.7)
    ldrDrago = tonemapDrago.process(hdrDebevec)
    ldrDrago = 3 * ldrDrago
    ldrDrago_8bit = np.clip(ldrDrago * 255, 0, 255).astype('uint8')
    cv2.imwrite("ldr-Drago.jpg", ldrDrago_8bit)


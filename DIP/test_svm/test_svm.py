import numpy as np
import cv2 


# size of the image after normalization
winSize = (64, 128)

# parameters for HOG Descriptor
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

# create HOG Descriptor
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)


descriptor_size = []
labels = []


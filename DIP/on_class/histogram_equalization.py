import numpy as np 
import matplotlib.pyplot as plt 
import cv2

np.random.seed(69)

# create an img 
# 6x6 img, 3-bit image
WIDTH = 6
HEIGHT = 6
BIT = 3
SIZE = (WIDTH,HEIGHT)
L = 2**BIT

img = np.random.randint(low=0, high= 2**BIT, size=SIZE)
print(img)


# step 1: calculate the histogram 
def cal_hist(img):
    # using an array and it 
    hist = np.zeros(2**BIT, dtype='uint8')
    for i in img:
        hist[i] += 1
    return hist


# step 2: calculate the PMF
# pmf = number of pixel value i / total number of pixels
def cal_pmf(hist):
    total_num_pixels = SIZE[0] * SIZE[1]
    return hist / total_num_pixels

hist = cal_hist(img)
pmf = cal_pmf(hist)

# step 3: calculate the CDF 
# cdf(0) = is pmf(0)
# cdf(n) = cdf(n-1) + pmf(n)

# for loop version
def cal_cdf(pmf):
    cdf = np.zeros(2**BIT)
    cdf[0] = pmf[0]

    for i in range(1, len(pmf)):
        cdf[i] = cdf[i-1] + pmf[i]
    
    return cdf

# recursion version 
# def cal_cdf_recursion(pmf, cdf, n):
#     if n <= 0:
#        cdf[0] = pmf[0]
#     return cdf[]
    


# step 4: map the pixel value
# new value = round(cdf * (L-1))

def new_pixel_value(cdf):
    return np.int64(np.round(cdf * (2**BIT - 1)))

    
    
# complete histogram equalization

def hist_equal(img):
    hist = cal_hist(img)
    pmf = cal_pmf(hist)
    cdf = cal_cdf(pmf)
    new_values = new_pixel_value(cdf)
    # mapping
    new_img = new_values[img]
        
    return new_img

new_img = hist_equal(img)

print(new_img)
    

    
# test with the real fucking one bitch 
def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BRG2RGB)
    
img = cv2.imread('exercise_1.png')

cv2.imshow(img)

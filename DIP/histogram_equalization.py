
import numpy as np

M = 10
N = 10

image = np.random.randint(0,255,size=(5,5))


# count 
pixel_values, count = np.unique(image.flatten(), return_counts = True)


histogram = dict(zip(pixel_values, count))

print(histogram)



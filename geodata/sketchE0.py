
# https://docs.opencv.org/trunk/d7/d4d/tutorial_py_thresholding.html

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

width = 255; height = width
img = np.zeros([width,height],dtype=np.uint8)
for y in range(height):
    for x in range(width):
        img[x,y] = x
print('img: ',type(img),img.shape)
status = cv.imwrite('gradient.png',img)


img = cv.imread('gradient.png',0)

# ok original 
trigger,out = (127,255)

# bad trigger,out = (0,93)
# erf... trigger,out = (94,255)
# trigger,out = (125,1) # 

ret,thresh1 = cv.threshold(img,trigger,out,cv.THRESH_BINARY)
ret,thresh2 = cv.threshold(img,trigger,out,cv.THRESH_BINARY_INV)
ret,thresh3 = cv.threshold(img,trigger,out,cv.THRESH_TRUNC)
ret,thresh4 = cv.threshold(img,trigger,out,cv.THRESH_TOZERO)
ret,thresh5 = cv.threshold(img,trigger,out,cv.THRESH_TOZERO_INV)
titles = ['Original Image','BINARY','BINARY_INV','TRUNC','TOZERO','TOZERO_INV']

print('thresh1 ',type(thresh1))
images0 = [img, thresh1, thresh2, thresh3, thresh4, thresh5]
images = [np.array(i,dtype=np.float) for i in images0]
for i in range(6):
    print("%20s %4f %4f"%(titles[i],np.amin(images[i]),np.amax(images[i])))
    plt.subplot(2,3,i+1)
    plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


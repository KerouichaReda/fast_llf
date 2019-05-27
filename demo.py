import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import llf
np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)
im= cv.imread('Lenna.png')
imrgb=cv.cvtColor(im,cv.COLOR_BGR2RGB)/255
img=cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255
img32=np.float32(img)
I=img32
I_ratio=imrgb/llf.repeat(img)
sigma=0.1
N=10
fact=5
im_e=llf.llf(img,sigma,fact,N)
im_ergb=llf.repeat(im_e)*I_ratio
plt.subplot(1,2,1)
imgplt=plt.imshow(imrgb)
plt.subplot(1,2,2)
imgplt=plt.imshow(im_ergb)
plt.show()


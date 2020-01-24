import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import llf

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

im= cv.imread('sjacek.jpg')
imrgb=cv.cvtColor(im,cv.COLOR_BGR2RGB)/255
img=cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255
img32=np.float32(img)
I=img32

I_ratio=imrgb/llf.repeat(img)

sigma=0.3
N=10
fact=0.5
t = time.time()
#Filtering

im_e=llf.llf(img,sigma,fact,N)
im_ergb=llf.repeat(im_e)*I_ratio
elapsed = time.time() - t
print(elapsed)
#plot the image
plt.subplot(1,2,1)
imgplt=plt.imshow(imrgb)
plt.subplot(1,2,2)
imgplt=plt.imshow(im_ergb)
plt.show()
wrt=cv.cvtColor(np.float32(im_ergb*255),cv.COLOR_RGB2BGR)
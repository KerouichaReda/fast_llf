import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import time
import llf

np.set_printoptions(precision=5)
np.set_printoptions(suppress=True)

im1= cv.imread('Images\LennaBL.tif')
im2= cv.imread('Images\LennaBW.tif')
I1=np.float32(cv.cvtColor(im1,cv.COLOR_BGR2GRAY)/255)
I2=np.float32(cv.cvtColor(im2,cv.COLOR_BGR2GRAY)/255)


sigma=0.1
N=4
fact=-0.75
t = time.time()
#Filtering
print(I1.shape)
print(I2.shape)
im_e=llf.xllf(I1,I2,sigma,fact,N)
im_ergb=llf.repeat(im_e)
elapsed = time.time() - t
print(elapsed)
#plot the image
im_con=np.concatenate((im1/255,im2/255,im_ergb),axis=1)
imgplt=plt.imshow(im_con)

plt.show()




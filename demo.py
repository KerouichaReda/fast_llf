import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import llf
size=5
n_levels=math.ceil(math.log(min(size,size))-math.log(2))+2
n_levels=3
sigma=0.8
fact=-1
N=5
im= cv.imread('Lenna.png')
imrgb=cv.cvtColor(im,cv.COLOR_BGR2RGB)/255
img=cv.cvtColor(im,cv.COLOR_BGR2GRAY)/255
img=np.float32(img)
imr=llf.repeat(img)
ratio=imrgb/imr
mr=llf.llf(img,sigma,fact,N)
mr_rp=llf.repeat(mr)
mrc=(mr_rp*ratio)

pyr_g=llf.gaussian_pyramid(img,n_levels,None)
pyr_g0=pyr_g[0];
pyr_g1=pyr_g[1];
pyr_g2=pyr_g[2];

pyr_l=llf.laplacian_pyramid(img,n_levels,None)
pyr_l0=pyr_g[0];
pyr_l1=pyr_g[1];
pyr_l2=pyr_g[2];

I=img
ref=0.5
I_remap=fact*(I-ref)*np.exp(-(I-ref)*(I-ref))/(2*sigma*sigma)



size=5
kernal = cv.getGaussianKernel((size),5)
kernal=kernal.dot(kernal.T)

frameBGR = cv.filter2D(I_remap, -1, kernal)


plt.subplot(1,2,1)
imgplt=plt.imshow(I_remap,cmap='gray')
plt.subplot(1,2,2)
imgplt=plt.imshow(mrc)
plt.show()
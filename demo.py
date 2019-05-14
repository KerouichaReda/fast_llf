import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import math
import llf
size=512
img= cv.imread('Lenna.png')
img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)/255
img=np.float32(img)
mat=np.float32(np.ones((size,size)))
child =np.array([1, 51200, 1, 51200])
N=3
size=5
nlev=4
mat1=np.zeros((16,16))
mat2=np.ones((8,8))
row=np.arange(0,16,2)
col=np.arange(0,16,2)
rr,cc=np.meshgrid(row,col)
mat1[rr,cc]=mat2
fil=llf.pyramid_filter()
m=llf.laplacian_pyramid(img,None,None)
#m=llf.upsample(img,fil,[0,1024,0,1024])
z=np.zeros((15,15))
imgplt=plt.imshow(m[1],cmap='gray')
plt.show()

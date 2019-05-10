import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
#import llf
#img= cv.imread('Lenna.png')
#sigma=0.2
#fact=5
N=9
#print(llf.llf(img,sigma,fact,N))

pyr=np.empty((3),dtype=object)
pyr[0]=np.zeros((16,16))
pyr[1]=np.zeros((8,8))
pyr[2]=np.zeros((4,4))

arr=np.arange(4)
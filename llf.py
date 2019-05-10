import cv2 as cv2
import numpy as np 
import math
import scipy
def llf(I,sigma,fact,N):
    (height,width,dim)=I.shape
    n_levels=math.ceil(math.log(min(height,width))-math.log(2))+2
    discretisation=np.linspace(0,1,N)
    discretisation_step=discretisation[1]

    return discretisation_step
def gaussian_pyramid(I,nlev,subwindow):
    (r,c)=I.shape
    if subwindow is None:
        subwindow=[1,r,1,c]
    if nlev is None:
        nlev=numlevels([r,c])
    pyr=np.empty((nlev),dtype=object)
    pyr[0]=I
    fil=pyramid_filter()
    for i in range(1,nlev+1)
        I=downsample(I,fil,subwindow)
        pyr[i]=I
    return = pyr

def numlevels(im_sz):
    min_d=min(im_sz)
    nlev=1
    while min_d>1:
        nlev=nlev+1
        min_d=(min_d+1)//2
    return min_d

def child_windows(parent,N):
    if N is None:
        N=1
    child =parent
    for k in range(N):
        child = (child+1)/2
        child[[0,2]]=math.ceil(child[[0,2]])
        child[[1,3]]=math.floor(child[[1,3]])

    return child

def downsample(I,filter,subwindow):
    (r,c)=I.shape
    if subwindow is None:
        subwindow=[1 r 1 c]
    subwindow_child=child_windows(subwindow)
    border_mode = 'reweighted'
    if border_mode=='reweighted':
        R=cv.filter2D(I,filter)
        Z=cv.filter2D(np.ones(I,shape),filter)
        R=R./Z
    reven=(subwindow(0)%2==0)*1
    ceven=(subwindow(2)%2==0)*1
    R=R[0+reven:2:r,0+ceven:2:c,:]


        

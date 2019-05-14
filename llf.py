import cv2 as cv
import numpy as np 
import math
import scipy
def llf(I,sigma,fact,N):
    (height,width,dim)=I.shape
    n_levels=math.ceil(math.log(min(height,width))-math.log(2))+2
    discretisation=np.linspace(0,1,N)
    discretisation_step=discretisation[1]
    input_gaussian_pyr=gaussian_pyramid(I,n_levels,None)
    output_laplace_pyr=laplacian_pyramid(I,n_levels,None)
    output_laplace_pyr[n_levels-1]=input_gaussian_pyr[n_levels-1]
    for ref in discretisation:
        I_remap=fact*(I-ref)*math.exp(I-ref)/(2*sigma*sigma)
        temp_laplace=laplacian_pyramid(I_remap,n_levels,None)
        for level in range(0:n_levels-1):
            output_laplace_pyr[level]=output_laplace_pyr[level]+math.abs(input_gaussian_pyr[level]-ref<discretisation)*temp_laplace[level]*(1-math.abs(input_gaussian_pyr[level]-ref)/discretisation_step)

    F=reconstruct_laplacian_pyramid(output_laplace_pyr)
    return F
def gaussian_pyramid(I,nlev,subwindow):
    (r,c)=I.shape
    if subwindow is None:
        subwindow=[1,r,1,c]
    if nlev is None:
        nlev=numlevels([r,c])    
    pyr=np.empty((nlev),dtype=object)
    pyr[0]=I
    fil=pyramid_filter()
    for i in range(1,nlev):
        I,sub=downsample(I,fil)
        pyr[i]=I
    return pyr
def numlevels(im_sz):
    min_d=min(im_sz)
    nlev=1
    while min_d>1:
        nlev=nlev+1
        min_d=(min_d+1)//2
    return nlev
def child_windows(parent,N=1):
    if N is None:
        N=1
    child =np.array(parent)
    for k in range(N):
        child = (child)/2
        child[0]=math.ceil(child[0])
        child[2]=math.ceil(child[2])
        child[1]=math.floor(child[1])
        child[3]=math.floor(child[3])

    return child
def downsample(I,filter):
    
    r,c=I.shape    
    subwindow=[0, r ,0 ,c]
    subwindow_child=child_windows(subwindow)
    border_mode = 'reweighted'
    if border_mode =='reweighted':
        R=cv.filter2D(I,cv.CV_32FC1,filter)
        Z=cv.filter2D(np.float32(np.ones(I.shape)),cv.CV_32FC1,filter)
        R=R/Z
    reven=(subwindow[0]%2==0)*1
    ceven=(subwindow[2]%2==0)*1
    row=np.arange(0+reven,r,2)
    col=np.arange(0+ceven,c,2)
    R=R[row][:]
    R=R[:,col]
    return (R,subwindow_child)
def pyramid_filter():
    f=np.asmatrix(np.array([0.05, 0.25, 0.4, 0.25, 0.05])).T
    f=f.dot(f.T)
    return f
def laplacian_pyramid(I,nlev,subwindow):
    (r,c)=I.shape
    if subwindow is None:
        subwindow=np.array([0,r,0,c])*1.0
    if nlev is None:
        nlev=numlevels([r,c])
    pyr=np.empty((nlev),dtype=object)
    fil=pyramid_filter()
    J=I
    for l in range(0,nlev-1):
        (I,subwindow_child)=downsample(I,fil)
        up=upsample(I,fil,subwindow)
        pyr[l]=J-up
        J=I
        subwindow=subwindow_child

    return pyr
def upsample(I,fil,subwindow):
    r=subwindow[1]-subwindow[0]
    c=subwindow[3]-subwindow[2]
    #k=size(I,3)
    reven=(subwindow[0]%2==0)
    ceven=(subwindow[2]%2==0)
    border_mode='reweighted'
    R=0
    if border_mode=='reweighted':
        
        R=np.zeros((int(r),int(c)))
        row=np.arange(0+reven,r,2)
        col=np.arange(0+ceven,c,2)
        
        row_r,col_c=np.meshgrid(row,col)
        
        R[col_c.astype(int),row_r.astype(int),]=I
        R=cv.filter2D(np.float32(R),cv.CV_32FC1,fil)
        Z=np.zeros((int(r),int(c)))        
        Z[row_r.astype(int),col_c.astype(int)]=np.ones(I.shape)
        Z=cv.filter2D(np.float32(Z),cv.CV_32FC1,fil)
        R=R/Z
    return R


        



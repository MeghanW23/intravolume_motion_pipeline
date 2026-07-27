#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 16:43:52 2020

@author: ch208071
"""
import numpy as np
import SimpleITK as sitk
import time
from scipy.sparse.linalg import LinearOperator
import scipy.sparse.linalg as splinalg
from joblib import Parallel, delayed

#
def Insert2dSliceinto3Dvolume(I1,I2,j):
    
    return sitk.Paste(I1,I2,I2.GetSize(),destinationIndex=[0,0,j])

def numpy4Dtositk(X_np,origin,direction,spacing,nv):
    
    Xtemp = []
    
    for k in range(0,nv):
        Xtemp.append(sitk.GetImageFromArray(X_np[k,:,:,:]))
        
    Y = sitk.JoinSeries(Xtemp)
    Y.SetOrigin(origin)
    Y.SetSpacing(spacing)
    Y.SetDirection(direction) 
    
    return Y

def resample(X_numpy,tfm,interpolator,origin,spacing,direction,default_value,sz):
    X = sitk.GetImageFromArray(X_numpy)
    X.SetOrigin(origin)
    X.SetSpacing(spacing)
    X.SetDirection(direction)
    R = sitk.ResampleImageFilter();
    R.SetTransform(tfm);
    R.SetOutputOrigin(origin)
    R.SetOutputDirection(direction)
    R.SetOutputSpacing(spacing)
    R.SetDefaultPixelValue(default_value)
    R.SetInterpolator(interpolator)
    R.SetSize(sz)
    return sitk.GetArrayFromImage(R.Execute(X))

def ApplyGaussianFilter(X,origin,direction,spacing):
    X_img = sitk.GetImageFromArray(X)
    X_img.SetOrigin(origin)
    X_img.SetDirection(direction)
    X_img.SetSpacing(spacing)
    G = sitk.DiscreteGaussianImageFilter()
    G.SetVariance([0,0,1.3]) # 
    GX_img = G.Execute(X_img);
    return sitk.GetArrayFromImage(GX_img)
    

def fwdMotionOperator(X,interpolator,origin,spacing,direction,params,params_ind,n1,n2,nsl,nv,slice_info,nslbysmsfac):
    
    sz = X.GetSize()
    X_np = sitk.GetArrayFromImage(X)
    sz_v = sz[0:3]
    default_value=0
    #var = 0
    #G = sitk.DiscreteGaussianImageFilter()
    #G.SetVariance([0,0,var]) # variance = 1.27*1.27=1.6129, this means FWHM = 2.35 sigma = 3 mm
    Yv= np.zeros([nv,nsl,n2,n1])
    for i in range(0,nv):
        Xv = X_np[i,:,:,:]
        Y_tfm = np.zeros([nsl,n2,n1])
        for j in range(0,nslbysmsfac):
            slices = slice_info[j]
            tfm = sitk.Euler3DTransform()
            tfm.SetParameters(params[params_ind[j]+ (i * nslbysmsfac),:])
            tfm.SetComputeZYX(True)
            inv_tfm = tfm.GetInverse()
            Im = resample(Xv,inv_tfm,interpolator,origin,spacing,direction,default_value,sz_v)
           # Im = ApplyGaussianFilter(Im,origin_v,direction_v,spacing_v) # applying gaussian smoothing
             #Xsl = Im[slices,:,:]
            Y_tfm[slices,:,:]= Im[slices,:,:]
        Yv[i,:,:,:] = Y_tfm    
     
    return Yv     
 
    
def bwdMotionOperator(X,interpolator,origin,spacing,direction,params,params_ind,n1,n2,nsl,nv,slice_info,nslbysmsfac):
    sz = X.GetSize()
    X_np = sitk.GetArrayFromImage(X)
#    origin_v = origin[0:3]
#    spacing_v = spacing[0:3]
#    direction_v = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0) # needs to be modified accordingly.
    sz_v = sz[0:3]
    
    # For Gaussian smoothing along the slice direction
    default_value=0
    #var = 0
    #G = sitk.DiscreteGaussianImageFilter()
    #G.SetVariance([0,0,var]) # variance = 1.27*1.27=1.6129, this means FWHM = 2.35 sigma = 3 mm
    Yv= np.zeros([nv,nsl,n2,n1])

    
    for i in range(0,nv):
        Xv = X_np[i,:,:,:]
        Y_tfm = np.zeros([nsl,n2,n1])
        for j in range(0,nslbysmsfac):
            slices = slice_info[j]
            temp =np.zeros([nsl,n2,n1])
            #Xsl = Xv[slices,:,:]
            tfm = sitk.Euler3DTransform()
            tfm.SetParameters(params[params_ind[j]+ (i * nslbysmsfac),:])
            tfm.SetComputeZYX(True)
            #inv_tfm = tfm.GetInverse()
            temp[slices,:,:] = Xv[slices,:,:]
            ## Need to define the transpose of gaussian filter operation.
            #temp = ApplyGaussianFilter(temp,origin_v,direction_v,spacing_v) # applying gaussian smoothing
            Y_tfm = Y_tfm + resample(temp,tfm,interpolator,origin,spacing,direction,default_value,sz_v)
        Yv[i,:,:,:] = Y_tfm       
       
    return Yv  

def fwdbwdMotionOperator(X,ind_i,interpolator,origin,spacing,direction,params,params_ind,nsl,n2,n1,slice_info,nslbysmsfac):
    sz = X.GetSize()

    
    Xv = sitk.GetArrayFromImage(X)
    
    default_value=0
    #var = 0
    #G = sitk.DiscreteGaussianImageFilter()
    #G.SetVariance([0,0,var]) # variance = 1.27*1.27=1.6129, this means FWHM = 2.35 sigma = 3 mm

    
    Y_tfm = np.zeros([nsl,n2,n1])
    for j in range(0,nslbysmsfac):
        slices = slice_info[j]
        temp =np.zeros([nsl,n2,n1])
        tfm = sitk.Euler3DTransform()
        tfm.SetParameters(params[params_ind[j]+ ind_i,:])
        tfm.SetComputeZYX(True)
        inv_tfm = tfm.GetInverse()
        Im = resample(Xv,inv_tfm,interpolator,origin,spacing,direction,default_value,sz)
            #Im = ApplyGaussianFilter(Im,origin_v,direction_v,spacing_v) # applying gaussian smoothing
        #Xsl = Im[slices,:,:]
        temp[slices,:,:] = Im[slices,:,:]
            #temp = ApplyGaussianFilter(temp,origin_v,direction_v,spacing_v) # applying gaussian smoothing
        Y_tfm = Y_tfm + resample(temp,tfm,interpolator,origin,spacing,direction,default_value,sz)

    return Y_tfm  


def f(Znp,ind_i,interpolator,origin,spacing,direction,params,params_ind,n1,n2,nsl,slice_info,nslbysmsfac,beta):
    Znp = np.reshape(Znp,[nsl,n2,n1],'F')
    Zimg = sitk.GetImageFromArray(Znp)
    lhs = fwdbwdMotionOperator(Zimg,ind_i,interpolator,origin,spacing,direction,params,params_ind,nsl,n2,n1,slice_info,nslbysmsfac) + beta*Znp # + 0.1*Znp
    #return lhs
    return np.reshape(lhs.flatten('F'),[n1*n2*nsl,1])

def solveZiusingCG(Z,rhs,ind_i,params,params_ind,origin,spacing,direction,interpolator,n1,n2,nsl,nv,slice_info,nslbysmsfac,sz,beta):
 
    g = lambda z:f(z,ind_i,interpolator,origin,spacing,direction,params,params_ind,n1,n2,nsl,slice_info,nslbysmsfac,beta)
    A1 = LinearOperator((sz,sz),matvec = g)
    z,info =  splinalg.cg(A1,rhs,x0 = Z, tol=1e-10,maxiter = 25, M = None,callback  = None, atol =1e-10 )
    
    return z

def functoParallelize(Zinit_i,rhs_i,ind_i,params,params_ind,origin,spacing,direction,interpolator,n1,n2,nsl,nv,slice_info,nslbysmsfac,sz,beta):
    
    Zi = np.reshape(Zinit_i,[n1*n2*nsl,1],'F')
    rhs_i = np.reshape(rhs_i,[n1*n2*nsl,1],'F') 
    Zr = solveZiusingCG(Zi,rhs_i,ind_i,params,params_ind,origin,spacing,direction,interpolator,n1,n2,nsl,nv,slice_info,nslbysmsfac,sz,beta)
    
    return Zr


def solveZsubproblem(Zinit,rhs,params,params_ind,origin,spacing,direction,interpolator,n1,n2,nsl,nv,slice_info,nslbysmsfac,sz,beta,Njobs):
    
    Zr = Parallel(n_jobs = Njobs,verbose=0)(delayed(functoParallelize)(Zinit[i,:,:,:],rhs[i,:,:,:],(i * nslbysmsfac),params,params_ind,origin,spacing,direction,interpolator,n1,n2,nsl,nv,slice_info,nslbysmsfac,sz,beta) for i in range(0,nv))
        
    return np.reshape(np.asarray(Zr),[nv,nsl,n2,n1],'F')   
        
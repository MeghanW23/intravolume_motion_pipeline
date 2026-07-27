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
    

def fwdMotionOperator(X,interpolator,origin,spacing,direction,params,n1,n2,nsl,nv):
    
    default_value=0
    sz = X.GetSize()
    sz_v = sz[0:3]
    X_np = sitk.GetArrayFromImage(X)
    Y = np.zeros([nv,nsl,n2,n1])
    for i in range(0,nv):
        Xv = X_np[i,:,:,:]
        tfm = sitk.Euler3DTransform()
        tfm.SetParameters(params[i,:])
        tfm.SetComputeZYX(True)
        inv_tfm = tfm.GetInverse()  
    #Y = sitk.Resample(X,X,inv_tfm,interpolator,0.0, X.GetPixelID())
        Y[i,:,:,:] = resample(Xv,inv_tfm,interpolator,origin,spacing,direction,default_value,sz_v)    
    return Y    
 
    
def bwdMotionOperator(X,interpolator,origin,spacing,direction,params,n1,n2,nsl,nv):

    sz = X.GetSize()
    sz_v = sz[0:3]
    default_value=0
    X_np = sitk.GetArrayFromImage(X)
    Y = np.zeros([nv,nsl,n2,n1])
    
    for i in range(0,nv):
        print('Function bwdMotionOperator() Volume ' + str(i) + ' of ' + str(nv))

        Xv = X_np[i,:,:,:]
        tfm = sitk.Euler3DTransform()
        tfm.SetParameters(params[i,:])
        tfm.SetComputeZYX(True)
        #Y[i,:,:,:]=sitk.Resample(Xv,Xv,tfm,interpolator,0.0, Xv.GetPixelID())
        Y[i,:,:,:] = resample(Xv,tfm,interpolator,origin,spacing,direction,default_value,sz_v)
        
    return Y
    
def fwdbwdMotionOperator(X,interpolator,origin,spacing,direction,par):
    

    sz = X.GetSize()
    sz_v = sz[0:3]
    default_value=0
    Xv = sitk.GetArrayFromImage(X)  
    tfm = sitk.Euler3DTransform()
    tfm.SetParameters(par)
    tfm.SetComputeZYX(True)
    inv_tfm = tfm.GetInverse()  
    Im = resample(Xv,inv_tfm,interpolator,origin,spacing,direction,default_value,sz_v)
    Y = resample(Im,tfm,interpolator,origin,spacing,direction,default_value,sz_v)
    
    return Y

 
def f(Znp,interpolator,origin,spacing,direction,par,n1,n2,nsl,beta):
    Znp = np.reshape(Znp,[nsl,n2,n1],'F')
    Zimg = sitk.GetImageFromArray(Znp)
    lhs = fwdbwdMotionOperator(Zimg,interpolator,origin,spacing,direction,par) + beta*Znp # + 0.1*Znp
    #lhs = (1/beta)*fwdbwdMotionOperator(Zimg,interpolator,origin,spacing,direction,par) + Znp 
    #return lhs
    return np.reshape(lhs.flatten('F'),[n1*n2*nsl,1])

def solveZiusingCG(Z,rhs,par,interpolator,origin,spacing,direction,n1,n2,nsl,nv,sz,beta):
 
    g = lambda z:f(z,interpolator,origin,spacing,direction,par,n1,n2,nsl,beta)
    A1 = LinearOperator((sz,sz),matvec = g)
    z,info =  splinalg.cg(A1,rhs,x0 = Z, tol=1e-10,maxiter = 25, M = None,callback  = None, atol =1e-10 )
    
    return z

def functoParallelize(Zinit_i,rhs_i,par,interpolator,origin,spacing,direction,n1,n2,nsl,nv,sz,beta):
    
    Zi = np.reshape(Zinit_i,[n1*n2*nsl,1],'F')
    rhs_i = np.reshape(rhs_i,[n1*n2*nsl,1],'F') 
    Zr = solveZiusingCG(Zi,rhs_i,par,interpolator,origin,spacing,direction,n1,n2,nsl,nv,sz,beta)
    
    return Zr


def solveZsubproblem(Zinit,rhs,params,origin,spacing,direction,interpolator,n1,n2,nsl,nv,sz,beta,Njobs):
    
    Zr = Parallel(n_jobs = Njobs,verbose=0)(delayed(functoParallelize)(Zinit[i,:,:,:],rhs[i,:,:,:],params[i,:],interpolator,origin,spacing,direction,n1,n2,nsl,nv,sz,beta) for i in range(0,nv))
        
    return np.reshape(np.asarray(Zr),[nv,nsl,n2,n1],'F')   
        
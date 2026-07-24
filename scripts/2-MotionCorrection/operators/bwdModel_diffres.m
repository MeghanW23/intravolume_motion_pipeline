% Y -  Acquired measurements in .nii format
% returns a large matrix
function Bt = bwdModel_diffres(Y,interpolator,ind,ilacq,params,origin,spacing,direction,n1,n2,nsl,nv,na,nb)

spacing_high = py.tuple([3,3,6,1.5]);

ex_v = py.SimpleITK.ExtractImageFilter();
ex_sl = py.SimpleITK.ExtractImageFilter();
Yv = py.list();
vol_size= py.list(int32([n1,n2,nsl,0]));
sl_size = py.list(int32([n1,n2,0]));
ex_v.SetSize(vol_size);
ex_sl.SetSize(sl_size);

% For Gaussian smoothing along the slice direction
var_Gauss = 0; %max_var = 1.6
G = py.SimpleITK.DiscreteGaussianImageFilter();
G.SetVariance([0,0,var_Gauss]);

tempind = [13,0,9,1,10,2,11,3,12,5,14,6,15,7,16,8,17,4]+1;% look-up could change with dataset
params_ind = repmat(tempind,1,2);
tic;
for i = 1:nv
    vol_index = py.list(int32([0,0,0,i-1])); % indexing here is done according to simpleITK
    ex_v.SetIndex(vol_index);
    Xv = ex_v.Execute(Y);
    
    
     % interpolator = py.SimpleITK.sitkNearestNeighbor;
     %interpolator = py.SimpleITK.sitkLinear;
        %interpolator = py.SimpleITK.sitkBSpline;
%        interpolator = py.SimpleITK.sitkCosineWindowedSinc;
    default_value = 0;
    
    Y_tfm = py.SimpleITK.Image(int32([n1,n2,nsl]),py.SimpleITK.sitkFloat64);
    Y_tfm.SetOrigin(Xv.GetOrigin());
    Y_tfm.SetSpacing(spacing_high);
    Y_tfm.SetDirection(Xv.GetDirection());
    
    for j = 1: nsl

        
        temp = py.SimpleITK.Image(int32([n1,n2,nsl]),py.SimpleITK.sitkFloat64);
        temp.SetOrigin(Xv.GetOrigin());
        temp.SetSpacing(spacing_high);
        temp.SetDirection(Xv.GetDirection());
        
        tfm = py.SimpleITK.Euler3DTransform();
        tfm.SetParameters(params(params_ind(j)+ ((i-1+115) * nsl/2),:));
        inv_tfm = tfm.GetInverse();
        
        ex_sl.SetIndex(py.list(int32([0,0,j-1])));
        Ysl = py.SimpleITK.JoinSeries(ex_sl.Execute(Xv));
        
        Ysl.SetSpacing(spacing_high);
        
        ref_sl = py.pyfuncs.Insert2dSliceinto3Dvolume(temp,Ysl,int32(j-1));
        
         Y_tfm = Y_tfm + py.SimpleITK.Resample(G.Execute(ref_sl),ref_sl,inv_tfm,interpolator,default_value);
       % Y_tfm = Y_tfm + py.SimpleITK.Resample(G.Execute(ref_sl),Xv,inv_tfm,interpolator,default_value);
       
    end
  Yv.append(Y_tfm);
end
toc;
clear Y
Ysm = py.SimpleITK.JoinSeries(Yv);
Ysm.SetOrigin(origin);
Ysm.SetSpacing(spacing);
Ysm.SetDirection(direction);
%py.SimpleITK.WriteImage(Ysm,'Ybwd_temp.nii.gz')
Bt_py = py.SimpleITK.GetArrayFromImage(Ysm);
btemp = numpytomatlab(Bt_py); 
% Here b should the matrix that reflects the acquisition
%mode
%%%%%%%%%%%%%%%%%%%%%%%% check here
b= btemp(:,:,ilacq,:);
Bt = zeros(na,nb);
Bt(ind) = b(:); 
end
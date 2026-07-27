function BtB = fwdbwdModel(X,interpolator,ind,ilacq,params,origin,spacing,direction,n1,n2,nsl,nv,na,nb)

X = reshape(X,na,nb);
b = zeros(n1,n2,nsl,nv);
btemp = reshape(X(ind),n1,n2,nsl,nv);
b(:,:,ilacq,:) = btemp; % The slices in each volume of b are arranged from 1 to nsl.
%clear X btemp

% Converting the large super-resolved matrix to the matrix having the same
% shape as the acquired one.
%X =  large2smallmatrix(Xtemp,ilacq,n1,n2,nsl,nv); %size(X) =  n1*n2*nsl x nv
X_numpy = matlabtonumpy(b);

% numpy to simple-ITK object
X_img = py.pyfuncs.numpy4Dtositk(X_numpy,origin,direction,spacing,int32(nv));
clear X_numpy
X_img.SetOrigin(origin);
X_img.SetSpacing(spacing);
X_img.SetDirection(direction);

ex_v = py.SimpleITK.ExtractImageFilter();
ex_sl = py.SimpleITK.ExtractImageFilter();
Yv = py.list();
vol_size= py.list(int32([n1,n2,nsl,0]));
sl_size = py.list(int32([n1,n2,0]));
ex_v.SetSize(vol_size)
ex_sl.SetSize(sl_size)

var_Gauss = 0;%max_var = 1.6
G = py.SimpleITK.DiscreteGaussianImageFilter();
G.SetVariance([0,0,var_Gauss]);

tempind = [13,0,9,1,10,2,11,3,12,5,14,6,15,7,16,8,17,4]+1;% look-up could change with dataset
params_ind = repmat(tempind,1,2);
for i =1:nv
    
    vol_index = py.list(int32([0,0,0,i-1]));
    ex_v.SetIndex(vol_index);
    Xv = ex_v.Execute(X_img);
    ref_img = Xv;
% %     interpolator = py.SimpleITK.sitkNearestNeighbor;
%     interpolator = py.SimpleITK.sitkLinear;
%      %interpolator = py.SimpleITK.sitkBSpline;
% %       interpolator = py.SimpleITK.sitkCosineWindowedSinc;
    default_value = 0;
    
    Y_tfm = py.SimpleITK.Image(int32([n1,n2,nsl]),py.SimpleITK.sitkFloat64);
    Y_tfm.SetOrigin(Xv.GetOrigin());
    Y_tfm.SetSpacing(Xv.GetSpacing());
    Y_tfm.SetDirection(Xv.GetDirection());
    
    for j = 1:nsl
        
        temp = py.SimpleITK.Image(int32([n1,n2,nsl]),py.SimpleITK.sitkFloat64);
        temp.SetOrigin(Xv.GetOrigin());
        temp.SetSpacing(Xv.GetSpacing());
        temp.SetDirection(Xv.GetDirection());
        
        tfm = py.SimpleITK.Euler3DTransform();
       %tfm.SetParameters(params(params_ind(j)+ ((i-1) * nsl/2),:));
       tfm.SetParameters(params(params_ind(j)+ ((i-1+115) * nsl/2),:)); % vol 116 to 151
        inv_tfm = tfm.GetInverse();
        
        Im_re = G.Execute(py.SimpleITK.Resample(Xv,ref_img,tfm,interpolator,default_value));
        ex_sl.SetIndex(py.list(int32([0,0,j-1])))
        Ysl = py.SimpleITK.JoinSeries(ex_sl.Execute(Im_re));
        Ysl.SetSpacing(spacing);
        %ref_sl = Insert2Dsliceinto3Dvolume(temp,Ysl,j); % insertion happening in matlab array; j index corresponding to matlab
        ref_sl = py.pyfuncs.Insert2dSliceinto3Dvolume(temp,Ysl,int32(j-1));
        
           G_ref_sl = G.Execute(ref_sl);
    
%         destinationIndex=py.list(int32([0,0,j]));
%         ref_sl= py.SimpleITK.Paste(temp, Ysl, Ysl.GetSize(), destinationIndex);
        Y_tfm = Y_tfm + py.SimpleITK.Resample(G_ref_sl,G_ref_sl,inv_tfm,interpolator,default_value);
%         Y_tfm = Y_tfm + py.SimpleITK.Resample(G.Execute(ref_sl),Xv,inv_tfm,interpolator,default_value);
    end
    Yv.append(Y_tfm);
end
Y = py.SimpleITK.JoinSeries(Yv);
Y.SetOrigin(origin);
Y.SetSpacing(spacing);
Y.SetDirection(direction);

%py.SimpleITK.WriteImage(Y,'Yfwdbwd.nii.gz')

BtB_py = py.SimpleITK.GetArrayFromImage(Y);
btemp = numpytomatlab(BtB_py); 
% Here b should the matrix that reflects the acquisition
%mode
b= btemp(:,:,ilacq,:);
BtB = zeros(na,nb);
BtB(ind) = b(:);% returns a 4D matrix

end
        
    








% X - super-resolved matlab array
% ind- indices corresponding to the measurements.
% ilacq -  slice acquisition order
% params - text file containing the motion parameters.
         % 1)First three elements are rotation (degrees) and the last three
         % are translation parameters
% origin,spacing, direction correspond to the original data.
% returns a matrix B, which contains slices affected by motion. 

function B = fwdModel_diffres(X,interpolator,ind,ilacq,params,origin,spacing,direction,na,nb,n1,n2,nsl,nv)

spacing_high = py.tuple([3,3,3,1.5]);

X = reshape(X,na,nb);
%Xtemp = zeros(na,nb); 
%Xtemp(ind) = X(ind);
b = zeros(n1,n2,nsl,nv);
btemp = reshape(X(ind),n1,n2,nsl,nv);
b(:,:,ilacq,:) = btemp; % The slices in each volume of b are arranged from 1 to nsl.
clear X btemp

% Converting the large super-resolved matrix to the matrix having the same
% shape as the acquired one.
%X =  large2smallmatrix(Xtemp,ilacq,n1,n2,nsl,nv); %size(X) =  n1*n2*nsl x nv

X_numpy = matlabtonumpy(b);

% numpy to simple-ITK object
X_img = py.pyfuncs.numpy4Dtositk(X_numpy,origin,direction,spacing,int32(nv));
clear X_numpy
X_img.SetOrigin(origin);
X_img.SetSpacing(spacing_high);
X_img.SetDirection(direction);

ex_v = py.SimpleITK.ExtractImageFilter();
ex_sl = py.SimpleITK.ExtractImageFilter();
Yv = py.list();

vol_size= py.list(int32([n1,n2,nsl,0]));
sl_size = py.list(int32([n1,n2,0]));
ex_v.SetSize(vol_size)
ex_sl.SetSize(sl_size)

var_Gauss = 0; %max_var = 1.6
G = py.SimpleITK.DiscreteGaussianImageFilter();
G.SetVariance([0,0,var_Gauss]);

tempind = [13,0,9,1,10,2,11,3,12,5,14,6,15,7,16,8,17,4]+1;% look-up could change with dataset
params_ind = repmat(tempind,1,2);
for i =1:nv
    
    Ysl = py.list();
    vol_index = py.list(int32([0,0,0,i-1]));
    ex_v.SetIndex(vol_index);
    Xv = ex_v.Execute(X_img);
    
    ref_img = Xv;
%      interpolator = py.SimpleITK.sitkNearestNeighbor;
%    interpolator = py.SimpleITK.sitkLinear;
    %interpolator = py.SimpleITK.sitkBSpline;
%      interpolator = py.SimpleITK.sitkCosineWindowedSinc;
     
    default_value = 0;
    
    for j = 1:nsl

        tfm = py.SimpleITK.Euler3DTransform();
        tfm.SetParameters(params(params_ind(j)+ ((i-1+115) * nsl/2),:));

%         Im_re1 = G.Execute(ReSampleVolume(Xv,tfm,origin_v,spacing_v,direction_v,sz));
        Im_re = G.Execute(py.SimpleITK.Resample(Xv,ref_img,tfm,interpolator,default_value));
        
        ex_sl.SetIndex(py.list(int32([0,0,j-1])))
        Ysl.append(ex_sl.Execute(Im_re)) 
    end
    Yv.append(py.SimpleITK.JoinSeries(Ysl,Xv.GetOrigin{3},Xv.GetSpacing{3})); 
end
Y = py.SimpleITK.JoinSeries(Yv); 
Y.SetOrigin(origin);
Y.SetSpacing(spacing);
Y.SetDirection(direction);

%py.SimpleITK.WriteImage(Y,'Yfwd.nii.gz')

B_py = py.SimpleITK.GetArrayFromImage(Y);
B = numpytomatlab(B_py); % returns a 4D matrix
% B = B(:);
end
        
    








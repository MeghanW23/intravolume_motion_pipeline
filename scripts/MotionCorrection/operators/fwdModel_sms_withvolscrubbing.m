% X - super-resolved matlab array
% ind- indices corresponding to the measurements.
% ilacq -  slice acquisition order
% params - text file containing the motion parameters.
         % 1)First three elements are rotation (degrees) and the last three
         % are translation parameters
% origin,spacing, direction correspond to the original data.
% returns a matrix B, which contains slices affected by motion. 

function B = fwdModel_np(X,interpolator,ind,ilacq,params,origin,spacing,direction,na,nb,n1,n2,nsl,nv)

X = reshape(X,na,nb);
%Xtemp = zeros(na,nb); 
%Xtemp(ind) = X(ind);
b = zeros(n1,n2,nsl,nv);
btemp = reshape(X(ind),n1,n2,nsl,nv);
b(:,:,ilacq,:) = btemp; % The slices in each volume of b are arranged from 1 to nsl.
clear X btemp

X_numpy = matlabtonumpy(b);
%params_py = matlabtonumpy(params);

% numpy to simple-ITK object
X_img = py.pyfuncs.numpy4Dtositk(X_numpy,origin,direction,spacing,int32(nv));
clear X_numpy
% X_img.SetOrigin(origin);
% X_img.SetSpacing(spacing);
% X_img.SetDirection(direction);

Yv = py.pyfuncs.fwdMotionOperator(X_img,interpolator,origin,spacing,params,int32(nsl),int32(nv));
Y = py.pyfuncs.numpy4Dtositk(Yv,origin,direction,spacing,int32(nv));
% Y.SetOrigin(origin);
% Y.SetSpacing(spacing);
% Y.SetDirection(direction);

%py.SimpleITK.WriteImage(Y,'Yfwd.nii.gz')

B_py = py.SimpleITK.GetArrayFromImage(Y);
B = numpytomatlab(B_py); % returns a 4D matrix
% B = B(:);
end
        
    








function BtB = fwdbwdModel_np_v2(X,interpolator,ind,ilacq,params,origin,spacing,direction,n1,n2,nsl,nv,na,nb)

X = reshape(X,na,nb);
b = zeros(n1,n2,nsl,nv);
btemp = reshape(X(ind),n1,n2,nsl,nv);
b(:,:,ilacq,:) = btemp; % The slices in each volume of b are arranged from 1 to nsl.
%clear X btemp
X_numpy = matlabtonumpy(b);

% numpy to simple-ITK object
X_img = py.pyfuncs.numpy4Dtositk(X_numpy,origin,direction,spacing,int32(nv));
clear X_numpy
% X_img.SetOrigin(origin);
% X_img.SetSpacing(spacing);
% X_img.SetDirection(direction);

Yv = py.pyfuncs.fwdbwdMotionOperator(X_img,interpolator,origin,spacing,params,int32(nsl),int32(nv));
Y = py.pyfuncs.numpy4Dtositk(Yv,origin,direction,spacing,int32(nv));
% Y.SetOrigin(origin);
% Y.SetSpacing(spacing);
% Y.SetDirection(direction);

%py.SimpleITK.WriteImage(Y,'Yfwdbwd.nii.gz')

BtB_py = py.SimpleITK.GetArrayFromImage(Y);
btemp = numpytomatlab(BtB_py); 
% Here b should the matrix that reflects the acquisition
%mode
b= btemp(:,:,ilacq,:);
BtB = zeros(na,nb);
BtB(ind) = b(:);% returns a 4D matrix

end
        
    








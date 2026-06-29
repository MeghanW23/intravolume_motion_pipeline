% Y -  Acquired measurements in .nii format
% returns a large matrix
function Bt = bwdModel_np_v2(Y,interpolator,ind,ilacq,params,origin,spacing,direction,n1,n2,nsl,nv,na,nb)

Yv  =py.pyfuncs.bwdMotionOperator(Y,interpolator,origin,spacing,params,int32(nsl),int32(nv));
Ysm = py.pyfuncs.numpy4Dtositk(Yv,origin,direction,spacing,int32(nv));
% Ysm.SetOrigin(origin);
% Ysm.SetSpacing(spacing);
% Ysm.SetDirection(direction);

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
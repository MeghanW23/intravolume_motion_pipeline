% Y -  Acquired measurements in .nii format
% returns a large matrix
function Bt = bwdModel_ss_np(Y,interpolator,ind,ilacq,params,origin,spacing,direction,origin_4d,spacing_4d,direction_4d,n1,n2,nsl,nv,na,nb)

Yv  =py.pyfuncs_ss.bwdMotionOperator(Y,interpolator,origin,spacing,direction,params,int32(nsl),int32(nv));
Ysm = py.pyfuncs_ss.numpy4Dtositk(Yv,origin_4d,direction_4d,spacing_4d,int32(nv));
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
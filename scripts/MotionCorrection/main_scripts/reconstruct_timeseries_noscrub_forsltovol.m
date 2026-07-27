%function [X,X_img,opt] = reconstruct_timeseries_noscrub(fmri_fname,opt)
function [X,X_img,opt] = reconstruct_timeseries_noscrub_forsltovol(Y_img,opt)
% *_np means the variable is a numpy array*
% *_img refers to a simple ITK object *
disp("Starting function reconstruct_timeseries_noscrub_forsltovol")
%% Load input fMRI time series
origin_4d = Y_img.GetOrigin();
spacing_4d = Y_img.GetSpacing();
direction_4d=  Y_img.GetDirection();
origin = origin_4d(1:3);
spacing = spacing_4d(1:3);
direction = direction_4d(1:3) + direction_4d(5:7) + direction_4d(9:11);
%Iorig_old = numpytomatlab(py.SimpleITK.GetArrayFromImage(Y_img));
Iorig = permute(double(py.SimpleITK.GetArrayFromImage(Y_img)),[3,4,2,1]);
[n1,n2,nsl,nvs] = size(Iorig);%% dimensions of the data

%vol_start =  opt.vol_start;
vol_start = 1;
nv = nvs-vol_start+1;
sms_fac = opt.sms_fac;
% n = nsl/sms_fac;
%I1 = Iorig(:,:,:,1);
I1 = opt.I1;
slice_info  = opt.slice_info;
params_ind = opt.params_ind;
nslbysmsfac = opt.nslbysmsfac;
%% Load input fmri time series without background
% Ynobg_img = py.SimpleITK.ReadImage(fmri_fname_bgremoved,py.SimpleITK.sitkFloat64);
% I_nobg = numpytomatlab(py.SimpleITK.GetArrayFromImage(Ynobg_img));
% I_nobg = I_nobg(:,:,:,vol_start:nvs);
% X_nobg = reshape(I_nobg,[n1*n2*nsl,nv]);
% opt.ind_bg = ~any(X_nobg,2); % index value of 1 corresponds to background pixel.
%ind_bg = opt.ind_bg;
%% Initialize the time series: all volumes are initialized to the first volume.
Zinit = repmat(I1,[1,1,1,nv]);
%%
interpolator =  py.SimpleITK.sitkBSpline;
sz = int32(n1*n2*nsl);
%% Dimensions of the super resolved time series -  every column of this matrix corresponds to a volume
n12 = n1*n2;
na = n12*nsl;
nb = nsl*nv/sms_fac;
%%
params = opt.params;
%% Parameters
mu = opt.mu;
beta = opt.beta;
beta_fac = opt.beta_fac;
overall_maxIter = opt.overall_maxIter;
Njobs = opt.Njobs;
ts_start = opt.ts_start;
slice_acq_order = opt.slice_acq_order;
p = opt.p;
%%
opt.na = na;
opt.nb = nb;
%%
option = 'fulldata';  %% You don't need to change this option.
Mask = CreateSamplingMask(option,slice_acq_order,sms_fac,n1,n2,nsl,nv,[]);
Mask = sparse(Mask);
%%
i=1;
index =1;
Mk = cell(1,nsl);
while i<=na
    %     Masktemp(index,:) = Mask(i,:);
    Mk{index} = diag(Mask(i,:));
    i = i+n12;
    index = index+1;
end
clear index i
opt.ind = find(Mask); % indices corresponding to sampled locations.
ind = opt.ind;
clear Mask Msub_il Msub v
%% rhs for the z-sub problem 
BtY_np  =py.pyfuncs_forsltovol.bwdMotionOperator(Y_img,interpolator,origin,spacing,direction,params,params_ind,int32(n1),int32(n2),int32(nsl),int32(nv),slice_info,int32(nslbysmsfac));
%% 
Zr = Zinit;
%Zinit_np_old = matlabtonumpy(Zr);
Zinit_np = py.numpy.asarray(permute(Zr,[4,3,1,2]));
%%
cost=[];
for i =  1: overall_maxIter
    i
    %% X-subproblem
    Z = Zr(:,:,slice_acq_order,:); % slices arranged as per the acquisition
    b = Z(:);
    [X,s,epsilon] = SLR(b,Mk,opt);
    Xtemp = reshape(X(ind),n1,n2,nsl,nv);
    Xsamp = zeros(n1,n2,nsl,nv);
    Xsamp(:,:,slice_acq_order,:) = Xtemp; %% Samples in the measured locations are retrieved from X (super-resolved time series).
    clear Xtemp
    %% Z-subproblem 
    %rhs_zsub_old = BtY_np+beta*matlabtonumpy(Xsamp);
    rhs_zsub = BtY_np+beta*py.numpy.asarray(permute(Xsamp,[4,3,1,2]));
    if i==1
        Zr_np = py.pyfuncs_forsltovol.solveZsubproblem(Zinit_np,rhs_zsub,params,params_ind,origin,spacing,direction,interpolator,int32(n1),int32(n2),int32(nsl),int32(nv),slice_info,int32(nslbysmsfac),int32(sz),beta,int32(Njobs));
    else
        Zr_np = py.pyfuncs_forsltovol.solveZsubproblem(Zr_np,rhs_zsub,params,params_ind,origin,spacing,direction,interpolator,int32(n1),int32(n2),int32(nsl),int32(nv),slice_info,int32(nslbysmsfac),int32(sz),beta,int32(Njobs));
    end
     
    Zr  = numpytomatlab(Zr_np);
    %Zr  = permute(double(Zr_np),[3,4,2,1]);
  
    %% Cost computation
    thiscost = compute_cost_withbeta(Zr,Iorig,Xsamp,s,epsilon,p,mu,beta,interpolator,params,params_ind,origin,spacing,direction,origin_4d,spacing_4d,direction_4d,n1,n2,nsl,nv,slice_info,nslbysmsfac);
    cost = [cost,thiscost];
    %figure(5),plot(cost);drawnow;
%     nm  = Zr - Xsamp;
%     sum(nm(:).^2)
    clear Xsamp
    beta = beta*beta_fac; % Changing beta at the end of every iteration.
end
opt.cost = cost;
%% Super-resolved time series can be downsampled from any starting point.
Xsamp = X(:,ts_start:nsl/sms_fac:nb); 
Isamp = reshape(Xsamp,n1,n2,nsl,nv);
%X_numpy_old = matlabtonumpy(Isamp);
X_numpy = py.numpy.asarray(permute(Isamp,[4,3,1,2]));
opt.origin_4d = origin_4d;
opt.spacing_4d = spacing_4d;
opt.direction_4d = direction_4d;
X_img = py.pyfuncs_forsltovol.numpy4Dtositk(X_numpy,origin_4d,direction_4d,spacing_4d,int32(nv));
opt.Xsamp = Xsamp;
opt.X= X;
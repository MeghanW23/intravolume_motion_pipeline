function [Xs,vol_rem] = MainFunctionForInterpScrubbedData(X_img,opt)

ft = opt.ft;
SWD = opt.SWD;
thresh_forscrub = opt.thresh_forscrub;
lower_thresh = opt.lower_thresh;
higher_thresh = opt.higher_thresh;

% Read reconstructed time series without scrubbing
%I = numpytomatlab(py.SimpleITK.GetArrayFromImage(X_img));
I = permute(double(py.SimpleITK.GetArrayFromImage(X_img)),[3,4,2,1]);

[n1,n2,nsl,nv] = size(I);
n12 = n1*n2;
X = reshape(I,n1*n2*nsl,nv);
[na,nb] = size(X);
Mask = ones(size(X));
opt.na = na;
opt.nb = nb;
%% Determine the volumes to be scrubbed
[vol_i,~,~] = find(SWD>=thresh_forscrub);
vol_toscr = VolsToScrub(vol_i,nv,lower_thresh,higher_thresh);
vol_rem = setdiff(1:nv,vol_toscr);
Mask(:,vol_toscr)=0;

% if vol_toscr >= ft && vol_toscr <= nv-10
%     ft = ft + 5;
% end

j=1;
index =1;
Mk = cell(1,nsl);
while j<=na
    
    Mk{index} = diag(Mask(j,:));
    j = j+n12;
    index = index+1;
    
end
clear index j
ind = find(Mask);
opt.ind = ind;
clear Mask

b  = X(ind);
%% parameters
[Xs,~,~] = SLR(b,Mk,opt);
% keyboard




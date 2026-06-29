function [X,s,epsilon_f] = SLR(b,Mk,opt)
%% IRLS based algorithm to reconstruct X, given Z

ind = opt.ind;
%ilacq = opt.slice_acq_order;
sms_fac = opt.sms_fac;
ft = opt.ft;
beta = opt.beta;
mu = opt.mu;
p = opt.p;
eta = opt.eta;
ts1 = opt.timepoint1;
ts2 = opt.timepoint2;
ind_bg = opt.ind_bg;
tolerance = opt.tolerance;
maxIter = opt.maxIter;

n1 = opt.n1;
n2 = opt.n2;
nsl = opt.nsl;
nv = opt.nv;
na = opt.na;
nb = opt.nb;
n12 = n1*n2;
%% Defining forward and backward operators
A = @(x)fwd(x,ind,na,nb);
At =@(y)bwd(y,ind,na,nb);
%%
Atb = At(b);
X = repmat(mean(Atb,2),[1,nb]);
%% Extracting time series corresponding to foreground pixels
X_bgrem = X;
X_bgrem(ind_bg,:)=[];
%%
Atb = sparse(Atb);
Atbcell = cell(1,nsl);
for j=1:nsl
    Atbcell{j} = Atb((j-1)*n12+1:j*n12,:);
end
clear Atb
%%
Ir = speye(ft);
n=1;
blk = size(X_bgrem,1);
%%
%%% Set the IRLS parameters
q = 1-(p/2);
epsilon=1;
%epsmin = 1e-9; %minimum possible epsilon value
%%
mubybeta = mu/beta;
d = [0:ft-1]';
k = nb-ft+1;

cost = [];
term1data= [];
term2reg =[];
for irlsiter=1:maxIter
    X_prev= X;
    %% Update X
    %tic;
    for i=1:blk
        Xt{i} = X_bgrem((i-1)*n+1:i*n,:).';
    end
    %toc;
    
    % Update W (weight matrix)
    tic;
    gramR  = zeros(ft);
    parfor l = 1:blk
        HankX = (im2colstep(Xt{l},[ft,1],[1,1]));
        gramR = gramR + (HankX*HankX');
    end
    clear l HankX Xt
    [V,S] = eig(gramR+epsilon*Ir);
    s = abs(diag(S));
   %figure(1);plot(s.^(0.5));title('s values');drawnow;
    clear S gramR
    alpha = s.^(-q/2);
    Q = V*diag(alpha);
    toc;
    %% Precomputing
    % Sum over all ft, Tqi'*Tqi; Tqi is toeplitz matrix formed from qi
    Sp = spalloc(nb,nb,2*ft);
    fcn = @plus;
    tic;
    parfor i1=1:ft
        vrep = repmat(Q(:,i1),[1,k]);
        tempdiag = spdiags(vrep.',d,k,nb);
        Sp= fcn(Sp,(tempdiag.'*tempdiag));
    end
    Sp = mubybeta*Sp;
    toc;
%     %% testing 
%     Sp_2 = spalloc(nb,nb,2*ft);
%     tic;
%     temp_rep=repmat(Q(:,1),[1,k]);
%     temp_rep_diag = (spdiags(temp_rep.',d,k,nb))';
%     ind_temp_rep_diag = find(temp_rep_diag );%temp_rep_diag~=0; 
%     parfor i1=1:ft
%         tempdiag_2 = sparse(nb,k);
%         vrep = repmat(Q(:,i1),[1,k]);
%         tempdiag_2(ind_temp_rep_diag) = vrep(:);
%         Sp_2= fcn(Sp_2,(tempdiag_2*tempdiag_2'));
%     end
%     Sp_2 = mubybeta*Sp_2;
%     toc;
%%
    tic;
    parfor i2 = 1:nsl
        %          X((i2-1)*n12+1:i2*n12,:) = Atb((i2-1)*n12+1:i2*n12,:)/(Sp+Mk{i2});
        X_timeseries_permuted(:,:,i2) = full(Atbcell{i2}/(Sp+Mk{i2}));
    end
    X = reshape(permute(X_timeseries_permuted,[1,3,2]),na,nb);
    toc;
    
    clear Xtemp
    X = X.*(X>0); % retaining only non-negative values
    
    %% Uncomment for cost computation. 
    term1 = b-A(X);
    if p == 0
        schatten = 0.5*sum(log(abs(s-epsilon)));
    else
        schatten = sum(abs((s-epsilon)).^(p/2));
    end
    thiscost = sum(term1(:).^2) + mubybeta*schatten;
    cost = [cost,thiscost];
   %figure(2);plot(cost);
    %%
    epsilon_f = epsilon;
    epsilon = epsilon/eta;
    
    %% Plotting reconstructed timeseries at two voxels. Uncommment if not
    % needed.
    %figure(3); plot(X(ts1,:)); drawnow;
   %figure(4);plot(X(ts2,:));drawnow;
    %% Computing the norm of the difference between two successive iterates.
    norm_er = (norm(X-X_prev,'fro')/norm(X_prev,'fro'));
    if  norm_er <=tolerance
        break;
    end
    X_bgrem = X;
    X_bgrem(ind_bg,:)=[];
end
end

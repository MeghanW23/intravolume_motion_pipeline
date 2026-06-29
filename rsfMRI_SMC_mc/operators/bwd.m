function At = bwd(y,ind,na,nb)
% X = gpuArray.zeros(na,nb);
X = zeros(na,nb);
% X = sparse(na,nb);
X(ind) = y;
At = X;
end
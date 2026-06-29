function AtA = bwdfwd(z,ind,na,nb)
% X = gpuArray.zeros(na,nb);
X = zeros(na,nb);
% X = sparse(na,nb);
z = reshape(z,na,nb);
X(ind) = z(ind);
AtA = X;
end
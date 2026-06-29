function A = fwd(x,ind,na,nb)
x = reshape(x,na,nb);
A = x(ind);
end
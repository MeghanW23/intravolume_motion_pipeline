% Y -  matlab array
% X - numpy object
function X = matlabtonumpy(Y)
sz = size(Y);
Y  = reshape(Y,sz);
Xtemp = py.numpy.array(Y(:).');
sz = size(Y);
ax = 0:length(sz)-1;
temp = ax(end); ax(end)=ax(end-1); ax(end-1) = temp; % dimension [a,b,c,d] ---> [a,b,d,c] (tranpose)
ax = int32(ax);
sz = int32(fliplr(sz));

X = Xtemp.reshape(sz).transpose(ax);
 
%% This is just for 1D arrays
if sz(1) == 1 ||  sz(2) == 1
    X = py.numpy.squeeze(X);
end



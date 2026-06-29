% X is numpy object
% Y is matlab array of type double.
% Dimension greater than 4D, verify the code.
function Y = numpytomatlab(X)
if X.ndim == 0 % when numpy array contains just one elemen -  a naive solution
    Y = double(py.array.array('d',py.numpy.nditer(X)));
elseif X.size == 0
    Y = [];
else
    sz = cellfun(@double, cell(X.shape));
    if length(sz)==1
        Y = double(py.array.array('d',py.numpy.nditer(X)));
    elseif length(sz)==2
        %Y = reshape(double(py.array.array('d',py.numpy.nditer(X))),fliplr(sz));
        Y = reshape(double(py.array.array('d',py.numpy.nditer(X,pyargs('order','F')))),sz);
    else
        Y = reshape(double(py.array.array('d',py.numpy.nditer(X,pyargs('order','C')))),fliplr(sz));
        ax = 1:length(size(Y));
        Y = permute(Y,[ax(2),ax(1),ax(3):ax(end)]);
    end
end
    
%else
    

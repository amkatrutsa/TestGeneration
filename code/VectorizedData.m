function [ rowX ] = VectorizedData( X )
% Function makes the row from matrix X. The row contains of 
% the transpose columns of the matrix X.  
%
% Input:
% X - [m, n] - data set
%
% Output:
% rowX - [1, m * n] - the row with the same information as in the data set

[m, n] = size(X);
rowX = reshape(X, 1, m * n);

end


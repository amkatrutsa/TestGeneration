function [ res ] = Rsq_adj( X, y, w, par )
% Compute adjusted R squared.
%
% Input:
% X - [m, p] - design matrix ith shrinkage number of predictors
% y - [m, 1] - target vector
% w - [p, 1] - vector of parameters, getting from algorithm
% par - structure with additional parameters, empty
%
% Output:
% res - [1, 1] - value of adjusted R squared

if(isempty(X))
    res = Inf;
    return
end
m = size(X, 1);
p = sum(w ~= 0);
tss = sumsqr(y - mean(y));
rss = sumsqr(y - X * w);
res = 1 - (rss / (m - p - 1)) / (tss / (m - 1));

end


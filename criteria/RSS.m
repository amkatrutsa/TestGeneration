function [ rss ] = RSS( X, y, w, par)
% RSS function computes the residual sum of squares for linear model 
%
% Input:
% y - [m, 1] - target vector
% X - [m, n] - design matrix
% w - [m, 1] - vector parameters getting from optimize problem
%
% Output:
% rss - [1, 1] - residual sum of squares

if(isempty(X))
    rss = Inf;
    return 
end
rss = sumsqr(y - X * w);

end


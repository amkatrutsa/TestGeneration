function [ w ] = ElasticNet( X, y )
%ELASTICNET Summary of this function goes here
%   Detailed explanation goes here
[W, inf] = lasso(X, y, 'Alpha', 0.5);
[~, idx_min] = min(inf.MSE);
w = W(:, idx_min);

end


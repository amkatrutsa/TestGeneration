function [ w ] = Lasso( X, y)
%MYLASSO Summary of this function goes here
%   Detailed explanation goes here
[W, fitinfo] = lasso(X, y);
[~, idx_minMSE] = min(fitinfo.MSE);
w = W(:, idx_minMSE);
end


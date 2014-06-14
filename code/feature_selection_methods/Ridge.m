function [ w ] = Ridge( X, y, par )
%MYRIDGE Summary of this function goes here
%   Detailed explanation goes here
k = 0:0.0001:0.05; % See for limit value of k
W = ridge(y, X, k);
rss = zeros(1, size(W, 2));
for i = 1:size(W, 2)
    rss(i) = sumsqr(y - X * W(:, i));
end

[~, idx_min] = min(rss);
w = W(:, idx_min);
end


function [ idxFSM ] = IdxBestFSM( X, y, crit, alg, param )
% Function finds the index of the best feature selection method (FSM) 
% among the FSM from alg according to the criterion crit  
% 
% Input:
% X - [m, n] - data set
% y - [m, 1] - target vector
% crit - string - selected external criterion
% alg - {cell array} - the list of the considered FSM
% param - structure with additional parameters:
%           s_0 - [1, 1] - limit error
%           
% Output:
% idxFSM - [1, 1] - index of the best FSM from the list alg according to
% the criterion crit
%
% Copyright Alexandr Katrutsa (c) 07.2014

crit_res = zeros(1, length(alg));
for i = 1:length(alg)
    X_shrink = X;
    w = feval(alg{i}, X, y);
    idx_del = abs(w) < 10^(-5); %10^(-6)
    w(idx_del) = [];
    X_shrink(:, idx_del) = [];
    crit_res(i) = feval(crit, X_shrink, y, w, param);
end
[~, idxFSM] = min(crit_res);
end
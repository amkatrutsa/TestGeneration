function [ TestError_logcond, TestError_numpar ] = AlgStable(alg, crit, ...
                                            objects, features, parameters)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
k = 0:0.01:0.97;
TestError_logcond = zeros(2 * length(alg), length(k));
TestError_numpar = zeros(2 * length(alg), length(k));
for i = 1:length(k)
    parameters.multpar = k(i);
    X = CreateData(objects, features, parameters);
    y = parameters.target;
    [ learn_X, test_X, learn_y, test_y ] = DivideLearnTest(X, y);
    for j = 1:length(alg)
        test_X_shrink = test_X;
        learn_X_shrink = learn_X;
        w = feval(alg{j}, learn_X, learn_y);
        idx_del = abs(w) < 10^(-6); %10^(-5)
        w(idx_del) = [];
        learn_X_shrink(:, idx_del) = [];
        test_X_shrink(:, idx_del) = [];
        TestError_logcond(2 * j, i) = sumsqr(test_y - test_X_shrink * w);
        TestError_logcond(2 * j - 1, i) = CondNumber(learn_X_shrink);
        TestError_numpar(2 * j, i) = TestError_logcond(2 * j, i);
        TestError_numpar(2 * j - 1, i) = size(w, 1);
    end
end
end


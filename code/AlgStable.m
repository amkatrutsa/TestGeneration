function [ TestError_logcond, TestError_numpar ] = AlgStable(alg, crit, ...
                                            objects, features, parameters)
% Function returns the pair test error, logarithm of the condition number and 
% the pair test error, number of the selected features for every FSM and 
% values of the multicollinearity parameter from 0 to 0.97 with step 0.05
%
% Input:
% alg - cell array with names of the algorithms to solve regression problem 
%                using feature selection for given design matrix X and target vector y
% crit - cell array with name of criterias using to test feature selection methods from alg
% objects - [1, 1] - number of the objects in generated data set, 
%                    the number of rows in the matrix X
% features - [1, 1] - number of features in generated data set,
%                     the number of columns in the matrix X,
%                     the dimension of the data
% parameters - structure with following parameters:
%              multpar - [1, 1] - parameter of the multicollinearity, 
%                                 if it equals 1, the data is full correlated
%              target - [objects, 1] - the target vector   
%              iter - [1, 1] - number of iteration for average matrix matAlgCrit
%
% Output:
% TestError_logcond - [2 * length(alg), length(k)] - matrix with test errors 
%                       and logarithm of the conditional numbers for every FSM
%                       and every value of k.
% TestError_numpar - [2 * length(alg), length(k)] - matrix with test errors 
%                       and number of the selected parameters for every FSM
%                       and every value of k.
%
% Copyright Alexandr Katrutsa (c) 05-06.2014

k = 0:0.05:0.97;
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
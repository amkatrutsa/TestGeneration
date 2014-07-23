function [ matAlgCrit, learn_error, test_error, num_parameters ] = ...
                        AlgCrit(alg, crit, objects, features, parameters)
% Function computes values of every considered criteria for result
% given every feature selection method
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
% matAlgCrit - [length(alg), length(crit)] - matrix contain the values, 
%                                            which are returned every criteria for 
%                                            every feature selection method 
%
% Copyright Alexandr Katrutsa (c) 05-06.2014

iter = parameters.iter;
matAlgCrit = zeros(length(alg), length(crit));
learn_error = zeros(1, length(alg));
test_error = zeros(1, length(alg));
num_parameters = zeros(1, length(alg));
for it = 1:iter
    parameters.target = randi(1.5 * objects, objects, 1);
    X = CreateData(objects, features, parameters);
    y = parameters.target;
    [ learn_X, test_X, learn_y, test_y ] = DivideLearnTest(X, y);    
    beta = lscov(learn_X, learn_y);
    rss_all = sumsqr(learn_y - learn_X * beta);
    tss_all = sumsqr(learn_y - mean(learn_y));
    par.s_0 = parameters.s_0;
    par.rss = rss_all;
    par.tss = tss_all;
    for i = 1:length(alg)
        learn_X_shrink = learn_X;
        test_X_shrink = test_X;
        w = feval(alg{i}, learn_X, learn_y);
        idx_del = abs(w) < 10^(-5); %10^(-6)
        w(idx_del) = [];
        learn_X_shrink(:, idx_del) = [];
        test_X_shrink(:, idx_del) = [];
        learn_error(i) = learn_error(i) + sumsqr(learn_y - learn_X_shrink * w);
        test_error(i) = test_error(i) + sumsqr(test_y - test_X_shrink * w);
        num_parameters(i) = num_parameters(i) + size(w, 1);
        for j = 1:length(crit)                
            matAlgCrit(i, j) = matAlgCrit(i, j) + feval(crit{j}, learn_X_shrink, ...
                                                    learn_y, w, par);
        end
        
    end
end
matAlgCrit = matAlgCrit / iter;
learn_error = learn_error / iter;
test_error = test_error / iter;
num_parameters = num_parameters / iter;
end
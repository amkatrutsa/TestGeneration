function [ matAlgCrit ] = AlgCrit(alg, crit, objects, features, parameters)
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
                
iter = parameters.iter;
matAlgCrit = zeros(length(alg), length(crit));
for it = 1:iter
    parameters.target = randi(1.5 * objects, objects, 1);
    X = CreateData(objects, features, parameters);
    y = parameters.target;
    len = sum(X.^2).^0.5;
    X = X ./ repmat(len, size(X, 1), 1);
    y = y ./ norm(y);
    beta = lscov(X, y);
    rss_all = sumsqr(y - X * beta);
    tss_all = sumsqr(y - mean(y));
    par.s_0 = parameters.s_0;
    par.rss = rss_all;
    par.tss = tss_all;
    for i = 1:length(alg)
        X_sh = X;
        w = feval(alg{i}, X, y);
        idx_del = abs(w) < 10^(-6);
        w(idx_del) = [];
        X_sh(:, idx_del) = [];
        for j = 1:length(crit)                
            matAlgCrit(i, j) = matAlgCrit(i, j) + feval(crit{j}, X_sh, y, w, par);
        end
    end
end
matAlgCrit = matAlgCrit / iter;
end
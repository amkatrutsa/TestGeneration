function [ mat_stability, s0 ] = Plotd_s0(alg, objects, features, parameters)
% Function computes the values of the stability criteria for wide range 
% critical error
% 
% Input:
% alg - cell array with names of the algorithms to solve regression problem 
%                using feature selection for given design matrix X and target vector y
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
% mat_stability - [length(alg), size(s0, 2)] - matrix contain the averaged values of stability criteria, 
%                                              for every feature selection method and every value of  
%                                              critical error s_0
% s_0 - [1, size(s0, 2)] - vector with values of critical error
%
% Copyright Alexandr Katrutsa (c) 05-06.2014

s0 = 0.01:0.05:2;
iter = parameters.iter;
mat_stability = zeros(length(alg), size(s0, 2));
for i = 1:length(alg)
    fprintf('Alg %s\n', alg{i});
    for j = 1:size(s0, 2)
        for k = 1:iter
            parameters.target = randi(1.5 * objects, objects, 1);
            X = CreateData(objects, features, parameters);
            y = parameters.target;
            len = sum(X.^2).^0.5;
            X = X ./ repmat(len, size(X, 1), 1);
            y = y ./ norm(y);
            X_sh = X;
            w = feval(alg{i}, X, y);
            idx_del = abs(w) < 10^(-6); % experienced cut off
            w(idx_del) = [];
            X_sh(:, idx_del) = [];
            par.s_0 = s0(j);
            mat_stability(i, j) = mat_stability(i, j) + stability(X_sh, y, w, par);
        end
    end
end
mat_stability = mat_stability / iter;
end


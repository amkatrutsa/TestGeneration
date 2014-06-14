function [ matVif, k] = PlotVif_k(alg, objects, features, parameters)
% Function computes VIF for different featureseoection methods and 
% parameters multicollinerity, which changes from 0 to 0.97 
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
% matVif - [length(alg), 97] - matrix with VIF for every feature selection method and 
%                              parameter multicollinearity 
% k - [1, 97] - vector with parameters of multicollinearity  

k = 0:0.01:0.97;
matVif = zeros(length(alg), size(k, 2));
for i = 1:length(alg)
    fprintf('Alg %s\n', alg{i});
    for j = 1:size(k, 2)
        parameters.multpar = k(j);
        parameters.target = randi(1.5 * objects, objects, 1);
        X = CreateData(objects, features, parameters);
        y = parameters.target;
        X = (X - repmat(mean(X), size(X, 1), 1)) ./ repmat(std(X), size(X, 1), 1);
        y = (y - mean(y)) / std(y);
        X_sh = X;
        w = feval(alg{i}, X, y);
        idx_del = abs(w) < 10^(-6); % experienced cut off
        w(idx_del) = [];
        X_sh(:, idx_del) = [];
        par = [];
        matVif(i, j) = Vif(X_sh, y, w, par);
    end
end
end


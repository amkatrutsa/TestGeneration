function [ d ] = stability( X, y, w, par )
% Function computes the number of features, 
% after their deleting the error function is less than par.s_0. 
% Deleting features is implemented through the Belsley procedure.     
%   
% Input:
% X - [m, p] - design matrix with shrinkage number of predictors
% y - [m, 1] - target vector
% w - [p, 1] - vector of parameters, getting from algorithm, 
%               which is tested
% par - structure - structure with additional parameters:
%       par.s_0 - [1, 1] - limit acccepted error rate
%
% Output:
% d - [1,1] - maximum number of possibly deleting features

if(isempty(X))
    d = Inf;
    return
end
s_0 = par.s_0;
idx_all_features = 1:max(size(w));
d = 0;
S = sumsqr(y - X * w);
% X_new = X;
while (S < s_0) && (size(idx_all_features, 2) > 1)
    idx_del = algBelsley(X, idx_all_features);
    idx_all_features(idx_all_features == idx_del) = [];
%     X_new(:, idx_del) = [];
%     w(idx_del) = [];
    d = d + 1;    
    S = sumsqr(y - X(:, idx_all_features) * w(idx_all_features));
end

end

function [ intIdxDelFeature ] = algBelsley( X, idxFeatures )
% Find index the worst feature, the most collinear, through the Belsley algorithm 
% [ intIdxDelFeature ] = algBelsley( structParam, structData )
% Input:
% structParam - struct of algorithms parameters with following parameters:
% 1) vecIdxFeatures - [1, k], vector of indeces selected features from feature will delete;
% 2) intNumFeatures - [1,1], total number of features;
% structData - struct of data parameters with following parameters:
% 1) matObjFeature - [n, p.intNumFeatures], matrix objects-features;

matObjFeature = X;
p = size(X, 2);
vecIdxCurrentFeatures = idxFeatures;
[~, S, V] =  svd(matObjFeature, 0);

lambda = diag(S);
% Ratio of largest singular value to all singular values
condind = repmat(S(1,1),p,1) ./ lambda;
condindCurrentFeature = condind(vecIdxCurrentFeatures);
[~, idxMaxCondInd] = max(condindCurrentFeature);
% variance decomposition proportions
phi_mat = (V'.*V') ./ repmat(lambda.^2,1,p);
phi = sum(phi_mat);
vdp = phi_mat ./ repmat(phi,p,1);
[~, intIdxDelFeature] = max(vdp(vecIdxCurrentFeatures(idxMaxCondInd), vecIdxCurrentFeatures), [], 2);
intIdxDelFeature = vecIdxCurrentFeatures(intIdxDelFeature);

% vecSingularValues = (diag(Lambda));
% vecConditionNumber = (vecSingularValues./max(vecSingularValues)).^(-1);
% % Find feature with maxima condition number.
% vecConditionNumberCurrentFeature = vecConditionNumber(vecIdxCurrentFeatures);
% intMaxConditionNumber = max(vecConditionNumberCurrentFeature);
% intIdxMaxConditionNumber = vecConditionNumberCurrentFeature == intMaxConditionNumber;
% if(sum(intIdxMaxConditionNumber) > 1)
%     intIdxMaxConditionNumber = find(intIdxMaxConditionNumber == 1, 1);
% end
% % Compute matrix of variance decomposition.
% matF = (V.^2) * diag(vecSingularValues.^(-2));
% matVarianceDecomposition = (diag(sum(matF, 2)'.^(-1)) * matF)';
% % Find index of column in row intIdxMaxConditionNumber with maxima variance
% % decomposition, it is the index of the worst feature.
% [~, intIdxDelFeature] = max(matVarianceDecomposition(intIdxMaxConditionNumber,vecIdxCurrentFeatures), [], 2);
% intIdxDelFeature = vecIdxCurrentFeatures(intIdxDelFeature);
end

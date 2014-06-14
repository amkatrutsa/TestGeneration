function [ intIdxDelFeature ] = algBelsley(structParam, structData)
% Find index the worst feature, the most collinear, through the Belsley algorithm 
%
% Input:
% structParam - struct of algorithms parameters with following parameters:
%               vecIdxFeatures - [1, k] - vector of indices selected features 
%               from feature will delete
%               intNumFeatures - [1, 1] - total number of features
% structData - structure of data parameters with following fields:
%              matObjFeature - [n, p.intNumFeatures] - matrix objects-features
%
% Output:
% intIdxDelFeature - [] - index of the most collinaer feature, which is needew to delete

matObjFeature = structData.matLearnObjFeatures;
vecIdxCurrentFeatures = structParam.vecIdxFeatures;
[~, Lambda, V] =  svd(matObjFeature);
vecSingularValues = (diag(Lambda));
vecConditionNumber = (vecSingularValues./max(vecSingularValues)).^(-1);
% Find feature with maxima condition number.
vecConditionNumberCurrentFeature = vecConditionNumber(vecIdxCurrentFeatures);
intMaxConditionNumber = max(vecConditionNumberCurrentFeature);
intIdxMaxConditionNumber = vecConditionNumberCurrentFeature == intMaxConditionNumber;
if(sum(intIdxMaxConditionNumber) > 1)
    intIdxMaxConditionNumber = find(intIdxMaxConditionNumber == 1, 1);
end
% Compute matrix of variance decomposition.
matF = (V.^2) * diag(vecSingularValues.^(-2));
matVarianceDecomposition = (diag(sum(matF, 2)'.^(-1)) * matF)';
% Find index of column in row intIdxMaxConditionNumber with maxima variance
% decomposition, it is the index of the worst feature.
[~, intIdxDelFeature] = max(matVarianceDecomposition(intIdxMaxConditionNumber,vecIdxCurrentFeatures), [], 2);
intIdxDelFeature = vecIdxCurrentFeatures(intIdxDelFeature);
end
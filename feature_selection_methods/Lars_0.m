function [weights] = LARS(features, target, params)
%
% Least angle regression algorithm
%   uses C_p Mallow's criterium
% %See algorithm descriptions at
% http://strijov.com/papers/krymova10modelselection.pdf 
%[featuresRating] = gmdh(features, target, params)
%
% Arguments
% Input
%   features - matrix of features, where rows are objects, and colums are feature vectors
%   target   - target feature vector
%    params                - model parameters
%    lars needn't parameters  - 
% Output
%   featuresRating - structure with rating for all features; has fields
%     isInformative - array of marks is particular feature informative (1) or not (0)
%     weight        - weight of particular feature id it is informative
%
% Example
%   nFeatures = 10;
%   nObjects = 30;
%   features = rand(nObjects, nFeatures);
%   target = rand(nObjects, 1);
%   params = [];
%   lars(features, target, params)
% 
% See also
% TODO
% 
% Revisions
% Supervisor: Krymova E., Date: 15.01.2010 E-mail ekkrym@gmail.com

X = features;
y = target;

[n,p] = size(X);
muA = zeros(n,1);  % current y-estimate
beta = zeros(p,1); % current parameter vector
b = [];  % list of models
A = [];  %current set of features
Cp = []; % Mallow's criterium 

betaAll = lscov(X,y);
MSE = sumsqr(y - X*betaAll)/n;

for i = 1:p 
  c = X'*(y-muA);
  B = setdiff(1:p,A); % all indexes except indexes in A
  [C, idxC] = max(abs(c(B))); % find maximal value of correlation and corresponding index j of column in X
  idxC = B(idxC); % returning to indexes from X
  A = unique([A; idxC]);% add new index to the current set
  Sj  = sign(c(A)); %signes of correlations
  XA = X(:,A)*diag(Sj);
  G = pinv(XA'*XA);
  % G = (XA'*XA)^(-1);
  oA = ones(length(A),1);
  AA =(oA'*G*oA)^(-0.5);
  wA = AA*G*oA;
  uA = XA*wA; %unit vector
  a = X'*uA;
  % get the step value
  if i<p
      M = [(C-c(B))./(AA-a(B));(C+c(B))./(AA+a(B))]; 
      M(find(M<=0))=+Inf;
      gamma = min(M);
  else
      gamma  = C/AA;
  end
  muA = muA + gamma*uA; %update y-approximation
  beta(A) = beta(A) + gamma*diag(Sj)*wA; % update parameters vector
  b = [b;beta'];
  Cp(end+1) = sumsqr(y - X*beta)/MSE - 2*i + n; % Mallows criterium
end
[~,idx] = min(Cp);% find the best model
bestModel = b(idx,:);
weights = bestModel';

return


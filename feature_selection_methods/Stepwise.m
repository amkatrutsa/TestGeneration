function [ w ] = Stepwise( X, y )
%STEPWISE Summary of this function goes here
%   Detailed explanation goes here

w = stepwisefit(X, y, 'display', 'off');

end


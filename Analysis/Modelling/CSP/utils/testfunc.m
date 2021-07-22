function [outputArg1] = testfunc(inputArg1, varargin)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
p = inputParser;
pansenDefault = 'jojo';
hansiDefault = 'klaus';
% addRequired(p, 'inputArg1');
disp(p.Parameters)
pansenValFunc = @(x) ischar(x); % @(x,p) (ischar(x) && ~ismember(x, p.Parameters));
addParameter(p, 'pansen', pansenDefault, pansenValFunc);
addParameter(p, 'hansi', hansiDefault, @(x) ischar(x)) 
if length(varargin) > 0
    parse(p, varargin{:});
    disp(p.Results.hansi)
end
outputArg1 = inputArg1;
end


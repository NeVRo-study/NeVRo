
function [yMean, yCI95] = calc_mean_95conf(mat)

N = size(mat, 1);                                     
yMean = nanmean(mat, 1);                                    % Mean Of All Experiments At Each Value Of ‘x’
ySEM = std(mat, 1, 'omitnan')/sqrt(N);
CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
yCI95 = bsxfun(@times, ySEM, CI95(:)); 
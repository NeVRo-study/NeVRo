
% Prelim script to write out predi vs target tables

% load results:
load('../../')

res_sum = CSP_results.results;
subjects = {res_sum.participant};
idx = find(cellfun('isempty', subjects));
% delete empty rows:
res_sum(idx) = [];
subjects(idx) = [];

% init tables:
target = nan(size(subjects,2),180);
predict = target;


for i=1:size(res_sum,2)
    if isempty(res_sum(i).participant) 
        target(i,:) = nan; 
        predict(i,:) = nan;
        subjects(i) = {'NaN'};
        continue
    end
    stats_sum = res_sum(i).stats;
    folds = stats_sum.per_fold;
    for f=1:size(folds,2)
        target(i,folds(f).indices{2}) = folds(f).targ;
        prediction = folds(f).pred{2}(:,1) < folds(f).pred{2}(:,2);
        predict(i,folds(f).indices{2}) = prediction' ;
    end
end

% format to tables and write to file:
targ_table = array2table(target, 'RowNames', subjects);
pred_table = array2table(predict, 'RowNames', subjects);

writetable(targ_table, 'targetTableCSP.csv', 'WriteRowNames',true, 'WriteVariableNames',false);
writetable(pred_table, 'predictionTableCSP.csv', 'WriteRowNames',true, 'WriteVariableNames',false);

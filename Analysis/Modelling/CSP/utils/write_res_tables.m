
% Prelim script to write out predi vs target tables

% load results:

for cond = {'mov', 'nomov'}

    % load model output:
    load(['CSP_results_' cond{1} '.mat'])

    % load binarized ratings (i.e., ground truth):
    ratFolder = ['./class_bins/' cond{1} '/SBA/'];
    ratFiles = dir([ratFolder 'NVR_*']);
    ratings = [];
    for iFile = 1:length(ratFiles)
        tmp = strsplit(ratFiles(iFile).name, '_run');
        ratTab = readtable([ratFolder ratFiles(iFile).name]);
        ratings(iFile).ID = tmp{1};
        ratings(iFile).values = ratTab.class_aro;
    end
    
    res_sum = CSP_results.results;
    subjects = {res_sum.participant};
    idx = find(cellfun('isempty', subjects));
    % delete empty rows:
    res_sum(idx) = [];
    subjects(idx) = [];

    % init tables:
    target = nan(size(subjects,2),270);
    predict = target;
    pred_prob = target;


    for i=1:size(res_sum,2)
        if isempty(res_sum(i).participant) 
            target(i,:) = nan; 
            predict(i,:) = nan;
            pred_prob(i,:) = nan;
            subjects(i) = {'NaN'};
            continue
        end
        stats_sum = res_sum(i).stats;
        folds = stats_sum.per_fold;
        for f=1:size(folds,2)
            % if the time series fed to the model starts with a "low" or
            % "high" arousal epoch, we need to pad the result with one NaN
            % as the model cropped of the first & last epoch 
            
            subIdx = [contains({ratings.ID}, subjects)];
            valTab = [ratings(subIdx).values]';
            padIdx = valTab(:,1) ~= 2;
            targetIdx = valTab ~= 2;
            for row = 1:size(targetIdx,1)
                colIdx(row,:) = find(targetIdx(row,:));
            end
            
            epoch = colIdx(i,folds(f).indices{2} + padIdx(i));
            target(i,epoch) = folds(f).targ;
            prediction = folds(f).pred{2}(:,1) < folds(f).pred{2}(:,2);
            predict(i,epoch) = prediction' ;
            prob = folds(f).pred{2}(:,1);
            idx = prob > 0.5;
            prob(idx) = prob(idx) * -1;
            prob(~idx) = 1 - prob(~idx);
            pred_prob(i,epoch) = prob;
        end
    end

    % format to tables and write to file:
    targ_table = array2table(target, 'RowNames', subjects);
    pred_table = array2table(predict, 'RowNames', subjects);
    predProbs_table = array2table(pred_prob, 'RowNames', subjects);

    writetable(targ_table, ['targetTableCSP_' cond{1} '.csv'], ...
        'WriteRowNames',true, ...
        'WriteVariableNames',false);
    writetable(pred_table, ['predictionTableCSP_' cond{1} '.csv'], ...
        'WriteRowNames',true, ...
        'WriteVariableNames',false);
    writetable(predProbs_table, ['predictionTableProbabilitiesCSP_' cond{1} '.csv'], ...
        'WriteRowNames',true, ...
        'WriteVariableNames',false);
end
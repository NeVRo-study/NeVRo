

%% get data:
% paths:
rand_files = dir(); %get files in current dir to get link to folder;
path_orig = rand_files(1).folder;
mov_cond = 'nomov';
cropstyle = 'SBA';
path_data = [path_orig '/../../../Data/'];
path_dataeeg =  [path_data 'EEG/'];
path_summaries = [path_dataeeg, '08.8_CSP_3x10f_regauto_auc_smote_1.0cor/' mov_cond '/' cropstyle '/summaries/']; % '08.7_CSP_3x10f_reg_auc_smote_1.0cor__partest_deleteme/'
path_results = [path_orig '/../../../Results/Tables/'];

% if scorer==mcr: acc=1-loss
% if scorer==-auc: auc=0-(-auc)
offset = 0;
if contains(path_summaries, {'mcr', 'acc'})
    offset = 1;
    scorer = 'mcr';
elseif contains(path_summaries, 'auc')
    offset = 0;
    scorer = 'auc';
else
    warning("Don't know which scoring scheme was used. Default to 'auc'.")
end


% real accuracies:
fname = [path_summaries 'CSP_results.mat'];
data_real = load(fname, 'CSP_results');
accs_real = [];
sub_ids = {};
for i=1:length(data_real.CSP_results.results)
    if ~isempty(data_real.CSP_results.results(i).participant)
        
        accs_real(end+1,:) = offset - data_real.CSP_results.results(i).trainloss(:);
        sub_ids{end+1} = data_real.CSP_results.results(i).participant;
    end
end

% permutated accuracies:
if ~isempty(dir([path_summaries '/*_merged.mat'])) 
    fname = [path_summaries 'CSP_results_perm_merged.mat'];
    data_perm = load(fname, 'CSP_results_perm_merged');
    data_perm.CSP_results = data_perm.CSP_results_perm_merged;
else
    fname = [path_summaries 'CSP_results_perm.mat'];
    data_perm = load(fname, 'CSP_results');
end
accs_perm = [];
for i=1:length(data_perm.CSP_results.results)
    if ~isempty(data_perm.CSP_results.results(i).participant)
        accs_perm(end+1,:) = offset - data_perm.CSP_results.results(i).trainloss.perm(:);
    end
end


%% Plot overall distributions
figure
accs_perm_mean = mean(accs_perm, 2);
h_perm = histfit(accs_perm_mean, 26);
h_perm(1).FaceColor = 'blue'; 
h_perm(1).FaceAlpha = 0.6;
h_perm(2).Color = 'blue';
hold on

accs_real_mean = mean(accs_real, 2);
h_real = histfit(accs_real_mean, 26);
h_real(1).FaceColor = 'red'; 
h_real(1).FaceAlpha = 0.6;
h_real(2).Color = 'red';

% calc significance:
sign_levels = [0.05, 0.01, 0.001];
[~, p,ci,stats] = ttest(accs_real, accs_perm_mean, 'tail', 'right');
n_sigstars = sum(p < sign_levels);
xL = xlim;
yL = ylim;
text(median(xL),0.99*yL(2),repmat('*', 1, n_sigstars), 'HorizontalAlignment','center','VerticalAlignment','top')

ylabel('count')
xlabel(scorer)
title(['Avg. ' scorer ' per subject'])
leg = legend([h_perm(2), h_real(2)], 'shuffled labels', 'real labels');
title(leg, [cropstyle, '  -  ', mov_cond])

%%
    
% Calc single subject results:
accs_real_bc = repmat(accs_real, 1, size(accs_perm, 2));
diff = accs_perm >= accs_real_bc;
% calc sign. by MC method:
p_vals = (sum(diff, 2)+1)/(size(accs_perm, 2) + 1);
sig_p05 = sum(p_vals <0.05);

%% Print stats results:

% Single subject level:
fprintf("######## CONDITION: %s --- %s ########\n", mov_cond, cropstyle)
fprintf("### SINGLE SUBJECT LEVEL: ###\n")
fprintf("%i out of %i subjects significant (p < .05).\n", sig_p05, size(accs_real, 1))
% Group level level:
fprintf("### GROUP LEVEL: ###\n")
fprintf("M = %f, SD = %f \n", mean(accs_real), std(accs_real))
fprintf("Range: %f - %f\n", min(accs_real), max(accs_real))
fprintf("t(%i) = %f, p = %f\n", stats.df, stats.tstat, p)

% Extract results table:
results_mat = [accs_real, accs_perm_mean, p_vals];
results_tab = array2table(results_mat);
results_tab.Properties.VariableNames = {'roc-auc', 'roc-auc_permuted_mean', 'p_val'};
results_tab.subID = string(sub_ids)';
results_tab = results_tab(:, {'subID', 'roc-auc', 'roc-auc_permuted_mean', 'p_val'});
fname = [path_summaries, 'results_table_' mov_cond '_', cropstyle '.csv'];
writetable(results_tab, fname, 'Delimiter', ',');

%% write to results table across methods:
fname = [path_results 'results_across_methods_' mov_cond '_supplementary_analysis.csv'];
tmp = readtable(fname, 'ReadVariableNames', true, 'Delimiter', ',');
tmp.([cropstyle '_BLOCK_CSP_auc']) = accs_real;
tmp.([cropstyle '_BLOCK_CSP_perm_auc']) = accs_perm_mean;
tmp.([cropstyle '_BLOCK_CSP_Pvalue']) = p_vals;
writetable(tmp, fname, 'Delimiter', ',');


%% 
figure
% per subject:
for isub=1:size(accs_real, 1)
    subplot(4, 7, isub)
    h = histfit(accs_perm(isub,:));
    h(1).FaceColor = 'blue'; 
    h(1).FaceAlpha = 0.6;
    h(2).Color = 'blue';
    hold on
    real = xline(accs_real(isub), 'Color', 'red', 'LineWidth', 1.5);
    xlabel(scorer);
    ylabel('count');
    title(sub_ids{isub}, 'Interpreter','none')
    ylim([0 80])
    
    n_sigstars = sum(accs_real(isub) > prctile(accs_perm(isub,:), [95, 99, 99.9]));
    xL=xlim;
    yL=ylim;
    text(mean([accs_real(isub), accs_perm_mean(isub)]),0.99*yL(2),repmat('*', 1, n_sigstars), 'HorizontalAlignment','center','VerticalAlignment','top')
end
subplot(4, 7, isub+2)
plot(0, 0, 0, 0)
axis off
leg = legend('shuffled labels', 'real labels', 'Location', 'east');
title(leg, [cropstyle, '  -  ', mov_cond])
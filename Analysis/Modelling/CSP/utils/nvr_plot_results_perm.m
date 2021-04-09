

%% get data:
% paths:
rand_files = dir(); %get files in current dir to get link to folder;
path_orig = rand_files(1).folder;
mov_cond = 'nomov';
cropstyle = 'SBA';
path_data = [path_orig '/../../../Data/'];
path_dataeeg =  [path_data 'EEG/'];
path_summaries = [path_dataeeg, '08.7_CSP_3x10f_reg_mcr_smote_0.4cor/' mov_cond '/' cropstyle '/summaries/'];

% if scorer==mcr: acc=1-loss
% if scorer==-auc: auc=0-(-auc)
offset = 0;
if contains(path_summaries, 'mcr')
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
fname = [path_summaries 'CSP_results_perm.mat'];
data_perm = load(fname, 'CSP_results');
accs_perm = [];
for i=1:length(data_perm.CSP_results.results)
    if ~isempty(data_perm.CSP_results.results(i).participant)
        accs_perm(end+1,:) = offset - data_perm.CSP_results.results(i).trainloss.perm(:);
    end
end


%% Plot overall distributions
figure
accs_perm_mean = mean(accs_perm, 2);
histogram(accs_perm_mean, 26, 'Normalization','count', 'FaceColor','blue', 'FaceAlpha', 0.6)
hold on;
y = 0.3:0.001:0.8;
mu = mean(accs_perm_mean);
sigma = std(accs_perm_mean);
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
shuf = plot(y,f,'LineWidth',1.5, 'Color', 'blue')
hold on

accs_real_mean = mean(accs_real, 2);
histogram(accs_real, 26, 'Normalization','count', 'FaceColor', 'red', 'FaceAlpha', 0.6)
y = 0.3:0.001:0.8;
mu = mean(accs_real);
sigma = std(accs_real_mean);
f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
real = plot(y,f,'LineWidth',1.5, 'Color', 'red')
legend([real, shuf], 'real', 'shuffled')
ylabel(scorer);

figure
% per subject:
for isub=1:size(accs_real, 1)
    subplot(4, 7, isub)
    histogram(accs_perm(isub,:), 5, 'Normalization','pdf', 'FaceColor','blue', 'FaceAlpha', 0.6)
    hold on;
    [ymin, ymax] = bounds(accs_perm(isub,:));
    y = ymin-0.05:0.001:ymax+0.05;
    mu = mean(accs_perm(isub,:));
    sigma = std(accs_perm(isub,:));
    f = exp(-(y-mu).^2./(2*sigma^2))./(sigma*sqrt(2*pi));
    shuffled = plot(y,f,'LineWidth',1.5, 'Color', 'blue')
    hold on
    real = xline(mean(accs_real(isub,:)), 'Color', 'red', 'LineWidth', 1.5)
    ylabel(scorer);
    title(sub_ids{isub}, 'Interpreter','none')
end
subplot(4, 7, isub+1)
axis off
legend([shuffled, real], 'shuffled', 'real')
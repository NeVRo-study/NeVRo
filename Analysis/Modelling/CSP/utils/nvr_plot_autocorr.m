
auto_corr_lags = 50;
mov_cond = 'mov';
cropstyle = 'SBA';

% use the actual permutations used in the decoding script (not necessary normally)
use_actual_perms = false;

if strcmp(cropstyle,'SBA')
    len_rats = 270;
else 
    len_rats = 240;
end

rand_files = dir(); %get files in current dir to get link to folder;
path_orig = rand_files(1).folder;
path_data = [path_orig '/../../../Data/'];
path_dataeeg = [path_data, 'EEG/'];
path_in_eeg = [path_dataeeg '08.7_CSP_3x10f_reg_auc_smote_0.2cor/' mov_cond '/' cropstyle '/summaries/'];
data_perm = load([path_in_eeg, 'CSP_results_perm.mat']);
path_rats = [path_data, 'ratings/class_bins/', mov_cond, '/' cropstyle '/'];
data_orig = load([path_in_eeg, 'CSP_results.mat']);
n_rows = length(data_perm.CSP_results.results);
tmp_perms = [data_perm.CSP_results.results];
tmp_orig = [data_orig.CSP_results.results];
tmp2 = {tmp_perms(:).participant};
keep_idx = find(~cellfun(@isempty,tmp2));
rat_files = dir([path_rats '/*.txt'])
data_perm_clean = tmp_perms(keep_idx);
data_orig_clean = tmp_orig(keep_idx);
perm_targs = zeros(length(data_perm_clean), 500, len_rats);
orig_targs = zeros(length(data_perm_clean), len_rats);
auto_corrs_perm_180 = nan(length(data_perm_clean), 500, auto_corr_lags + 1);
auto_corrs_perm_10 = auto_corrs_perm_180;
auto_corrs_perm_5 = auto_corrs_perm_180;
auto_corrs_orig = zeros(length(data_perm_clean), auto_corr_lags + 1);
skips = [];
for i=1:length(data_perm_clean)
    data_orig_clean = rat_files(i).name;
    targs = csvread([path_rats, '/', data_orig_clean], 1, 1);
    auto_corrs_orig(i,:) = autocorr(targs, 'NumLags', auto_corr_lags);
    data_perm_180 = nvr_shuffle_eventtypes(targs', [1, 3], 180, 500, 0.2);
    data_perm_10 = nvr_shuffle_eventtypes(targs', [1, 3], 10, 500, 0.2);
    try
        data_perm_5 = nvr_shuffle_eventtypes(targs', [1, 3], 6, 500, 0.2);
    catch 
        disp('Problem');
        % data_perm_5 = nan(size(data_perm_10));
        % skips(end+1) = i;
        continue;
    end
        
    
    for j = 1:500
        if use_actual_perms 
            targs_10 = [data_perm_clean(i).stats.perm(j).targets];
            targs_180 = targs_10;
            targs_5 = targs_10;
        else
            targs_180 = data_perm_180(:,i);
            targs_10 = data_perm_10(:,i);
            targs_5 = data_perm_5(:,i);
        end
        perm_targs(i,j,:) = targs;
        auto_corrs_perm_180(i,j,:) = autocorr(targs_180, 'NumLags', auto_corr_lags);
        auto_corrs_perm_10(i,j,:) = autocorr(targs_10, 'NumLags', auto_corr_lags);
        auto_corrs_perm_5(i,j,:) = autocorr(targs_5, 'NumLags', auto_corr_lags);
    end
end

mean_acfs_perm_180 = squeeze(nanmean(auto_corrs_perm_180, 2));
mean_acfs_perm_10 = squeeze(nanmean(auto_corrs_perm_10, 2));
mean_acfs_perm_5 = squeeze(nanmean(auto_corrs_perm_5, 2));

[yMean_perm_180, yCI95_perm_180] = calc_mean_95conf(mean_acfs_perm_180);
[yMean_perm_10, yCI95_perm_10] = calc_mean_95conf(mean_acfs_perm_10);
[yMean_perm_5, yCI95_perm_5] = calc_mean_95conf(mean_acfs_perm_5);
[yMean_orig, yCI95_orig] = calc_mean_95conf(auto_corrs_orig);

figure
col_orig = plot(0:50, yMean_orig, 'red');
hold on
col_perm_180 = plot(0:50, yMean_perm_180, 'blue');
hold on
col_perm_10 = plot(0:50, yMean_perm_10, 'green');
hold on
col_perm_5 = plot(0:50, yMean_perm_5, 'yellow');
hold on 
plot(0:50, yCI95_orig + yMean_orig, '--', 'color', 'red' )
hold on
plot(0:50, yCI95_perm_180 + yMean_perm_180, '--', 'color', 'blue' )
hold on
plot(0:50, yCI95_perm_10 + yMean_perm_10, '--', 'color', 'green' )
hold on
plot(0:50, yCI95_perm_5 + yMean_perm_5, '--', 'color', 'yellow' )
ylabel('autocorrelation')
xlabel('lag')
leg = legend([col_orig, col_perm_180, col_perm_10, col_perm_5], 'real labels', 'random', '10 blocks', '6 blocks (250 iter)');
title(leg, [cropstyle, '  -  ', mov_cond])
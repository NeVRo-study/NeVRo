%% NeVRo_SPoC_SA

% This script computes SPoC between the SSD alpha-band filtered components and
% the continous arousal ratings of participants (z). _perm_ is used to
% denote the fact that p-values are computed outside of the SPoC functions
% and directly via permutation in the script. The focus here is on data
% (EEG and ratings) without break (SA).
%% Clean previous mess

clc
%clear all

%% Set paths
% NB: Add your general matlab and eeglab paths

% Open EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Paths
master_path = '../../NeVRo/';
addpath(master_path);

ssd_path = [master_path 'Data/EEG/07_SSD/'];
rat_path = [master_path 'Data/ratings/continuous/z_scored/']; 
spoc_path = [master_path 'Data/EEG/08.1_SPOC/']; 
results_path = [master_path 'Results/EEG/SPOC/']; 

% Folders
cond = {'nomov','mov'};

% Set Seed for reproducibility
rng(11); 

% Set Number of permutations
n_perm = 1000;

%% Load Data

for folder = 1:length(cond)
%for folder  = 1
     
rawDataFiles = dir([ssd_path cond{folder} '/SA/narrowband/*.set']);  %we specifcally use SA data

% NB: for the mov condition, S09, S24 and S27 are problematic because none of
% their SSD components passed our selection criteria 

% Initialize all the SPoC summary tables
%SPOC_Table_lambda= ones(length(rawDataFiles),2);
spoc_t = ones(length(rawDataFiles),3);
SPOC_Table_A= ones(32,length(rawDataFiles)+1);
SPOC_Table_W= ones(32,length(rawDataFiles)+1);

% Open up figure for topoplots
figure('units','normalized','outerposition',[0 0 1 1]);
hold on;

for isub = 1:length(rawDataFiles)
    
loadName = rawDataFiles(isub).name;
fileName = loadName(1:7);

% Load EEG file
[EEG,com]= pop_loadset([ssd_path cond{folder} '/SA/narrowband/' loadName]);
EEG.setname=fileName;
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%% Choose the correct ratings and the selected SSD components to be extracted:

% SSD Selected Components file
ssd_tab = readtable([ssd_path cond{folder} '/SSD_selected_components_' cond{folder} '.csv']); %import the selected components table
%ssd_sel =  str2num(cell2num(ssd_tab{str2num(fileName(6:end)),2})); all together
    
sel_sub = ssd_tab(str2num(fileName(6:end)),2); %select the correspondent participant's row
ssd_sel = str2num(cell2mat(table2array(sel_sub))); %convert selected row to list of numbers (indexes indicating components - Terrible sequence of nested functions but it works)

% Break the loop and start with a new participants if selected ssd
% components are less than <4
if (length(ssd_sel)<4) 
    continue 
end 

% Ratings
if exist([rat_path cond{folder} '/SA/' fileName '_run_1_comb_rat_z.txt'], 'file')
    rat_tab=readtable([rat_path cond{folder} '/SA/' fileName '_run_1_comb_rat_z.txt']);
else
    rat_tab=readtable([rat_path cond{folder} '/SA/' fileName '_run_2_comb_rat_z.txt']);
end
    
% Select the correct ratings
rat=rat_tab{:,2}; %Take the right coloumn of the (non) z-scored ratings
z=rat'; %call it 'z'
%save(['z.mat'],'z'); %if you want to use pop_spoc

% Select the correct SSD components and filters
ssd_comps_nosel = EEG.etc.SSD.W' * EEG.data; %first extract the full matrix of current (aka non-selected) SSD components

ssd_comps = ssd_comps_nosel(ssd_sel,:); %selected SSD components 
W_ssd = EEG.etc.SSD.W(:,ssd_sel); %selected W
A_ssd = EEG.etc.SSD.A(:,ssd_sel); %selected A
X_ssd = A_ssd * ssd_comps;  %new sensor space with selected SSD components only

EEG.data = X_ssd;  %needed if you want to use pop_spoc  

% Kill VEOG and HEOG before SPoC
% EEG = pop_select( EEG,'nochannel',{'HEOG' 'VEOG'});
% [ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%% SPoC structure preparation

% Load events
epochs=zeros(240,2); %SA

for i=1:240
    
    epochs(i,1)=i;
    epochs(i,2)=1;
    
end 

% Create 1 second epochs
EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], 'newname', EEG.setname, 'epochinfo', 'yes');

% Transform data in the right format for SPoC's original function 
X = permute(EEG.data, [2, 1, 3]);

% Define parameters of X that we will need for further processing
Te = length(X(:,:,1));  % number of samples per epoch
Nx = length(X(1,:,1));  % number of sensors

% Apply SPoC
%[W, A, lambda_values, p_values_lambda, Cxx, Cxxz, Cxxe] = spoc(X, z, 'n_bootstrapping_iterations',500);
[lambda_samples_end, z_shuff, W_perm, R, lambda_samples, lambda_samples_sign, r_samples,r_samples_sign, W, A, lambda_values, p_values_lambda, Cxx, Cxxz, Cxxe]  = spoc_corr(X, z, 'n_bootstrapping_iterations',n_perm);

%% Compute z_est (the estimated time course of the behavioural data - the univariate variable z - with the best set of weights W)
% AKA: projecting the data onto the best (most negative/last in our case) component

%% 1. Approach based on spoc_example: using X*W directly 
% s_est = zeros(Te, length(z));
% 
% for k=1:length(z)
%     s_est(:,k) = squeeze(X(:,:,k)) * W(:,end); % end to automatically take the last one ("most negative" lambda)
% end
% 
% p_est = var(s_est);

%% 2. Approach based on spoc function: using the trialwise covariance matrix
% computed from X and get_var_features.

p_est = get_var_features(W, Cxxe);
p_est = p_est(end,:); 

%% Compute correlation between z_est and z (in this case there's no training/test separation - might be overfitted)

% Spearman
rho  = corr(p_est', z', 'type', 'Spearman');

% Pearson
%r_tot = corrcoef(p_est', z');
%r = r_tot(1,2:end);

% Pearson
r = corr(p_est', z');

%% Permutations

% Let's compute p-values based on Spearmann correlation and not on lambdas

%% 1. First approach: classic
%rho_perm = zeros(1, n_perm);

% for p=1:n_perm
%     
%     W_curr = W_perm{p};  % current permuted W
%     
%     s_est_curr = zeros(Te, length(z));
% 
%     for kp=1:length(z)
%     s_est_curr(:,kp) = squeeze(X(:,:,kp)) * W_curr(:,end); %end to automatically take the last one ("most negative" lambda)
%     end
% 
% p_est_curr = var(s_est_curr);
% 
% % Spearman
% rho_perm(p) = corr(p_est_curr', z', 'type', 'Spearman');
%     
% end 

%% 2. Second approach: get_var_features

rho_perm = zeros(1, n_perm);
r_perm = zeros(1, n_perm);

for p=1:n_perm
    
    W_curr = W_perm{p};  % current permuted W
    z_shuffled = z_shuff{p};
    
    p_est_curr = get_var_features(W_curr, Cxxe);
    p_est_curr = p_est_curr(end,:); 
   
% Correlation
rho_perm(p) = corr(p_est_curr', z_shuffled', 'type', 'Spearman');
r_perm (p) = corr(p_est_curr', z_shuffled');
    
end 

% Compute average permuted Spearmann and store it
rho_perm_avg = mean(rho_perm);

% Compute average permuted Pearson and store it
r_perm_avg = mean(r_perm);


%% Compute p-value

% 1. With the "directional approach": how many permuted values are actually
% more negative than ours (in this case *always* the most negative
% component)

p_value_spear_neg = sum(rho_perm(:)<=rho)/n_perm;  %problema per componenti che during perm creano segni misti
p_value_pearson_neg = sum(r_perm(:)<=r)/n_perm;  %problema per componenti che during perm creano segni misti
p_value_lambda_neg = sum(lambda_samples_end(:)<=lambda_values(end))/n_perm;

% 2. With the Absolute approach (using the last component): results should be the same as the negtive
% ones IF lambda and/or corr values are negative. In the case of "mixed"
% signs they will differ of course
p_value_spear = sum(abs(rho_perm(:))>=abs(rho))/n_perm; 
p_value_pearson = sum(abs(r_perm(:))>=abs(r))/n_perm;  
p_value_lambda = sum(abs(lambda_samples_end(:))>=abs(lambda_values(end)))/n_perm;

% 3. SPoC function approach: using max abs value in each permutation (the
% null distribution might include values that do NOT pertain to the last
% component itself, since the idea is to always find the max abs value in
% all the permuted ones with the SPoC function.
%p_value_spear_max = sum(abs(rho_perm(:))>=abs(rho))/n_perm; 
%p_value_pearson_max = sum(abs(r_perm(:))>=abs(r))/n_perm;  

%p_value_lambda_max = sum(abs(lambda_samples(:))>=abs(lambda_values(end)))/n_perm;
p_value_lambda_max  = p_values_lambda(end);

%% Save Spatial Patterns

dlmwrite([spoc_path cond{folder} '/SA/' fileName '_A.csv'], A); % save spatial patterns
save([spoc_path cond{folder} '/SA/' fileName '_A.mat'], 'A'); %spatial patterns

%dlmwrite([spoc_path cond{folder} '/SA/' fileName '_A.csv'],comp_tc); %time course

%% Add everything to the general SPOC_results structure (one per condition - only considering the best SPOC component)

SPOC_results.chanlocs = EEG.chanlocs;
SPOC_results.results(isub).participant = fileName;
SPOC_results.results(isub).preprocessing = EEG.etc;
%SPOC_results.results(isub).SPOC_model.Cov = Cxx ;
%SPOC_results.results(isub).SPOC_model.Covz = Cxxz;
SPOC_results.results(isub).SPOC_model.Covtr = Cxxe;
SPOC_results.results(isub).SPOC_model.z_original = z;
SPOC_results.results(isub).SPOC_model.z = (z-mean(z(:)))./std(z(:));
SPOC_results.results(isub).SPOC_model.z_est = p_est;
SPOC_results.results(isub).SPOC_model.spearmann = rho;
SPOC_results.results(isub).SPOC_model.pearson = r;
SPOC_results.results(isub).SPOC_model.lambda = lambda_values;
SPOC_results.results(isub).SPOC_model.p_values_lambda = p_values_lambda;
SPOC_results.results(isub).stats.rho_abs = p_value_spear;
SPOC_results.results(isub).stats.rho_neg = p_value_spear_neg;
SPOC_results.results(isub).stats.r_abs = p_value_pearson;
SPOC_results.results(isub).stats.r_neg = p_value_pearson_neg;
SPOC_results.results(isub).stats.lambda_max = p_value_lambda_max;
SPOC_results.results(isub).stats.lambda_abs = p_value_lambda;
SPOC_results.results(isub).stats.lambda_neg = p_value_lambda_neg;
SPOC_results.results(isub).weights.SPOC_A = A(:,end);
SPOC_results.results(isub).weights.SPOC_W = W(:,end);
SPOC_results.results(isub).weights.SSD_W_sel = W_ssd;
SPOC_results.results(isub).weights.SSD_A_sel = A_ssd;

%% Update summary structure

struct_summ(isub).ID = fileName;
struct_summ(isub).lambda = lambda_values(end);
struct_summ(isub).rho = rho;
struct_summ(isub).r = r;

struct_summ(isub).rho_perm_avg = rho_perm_avg;
struct_summ(isub).r_perm_avg = r_perm_avg;

struct_summ(isub).pval_lambda_max=  p_value_lambda_max;
struct_summ(isub).pval_lambda_abs=  p_value_lambda;
struct_summ(isub).pval_lambda_neg=  p_value_lambda_neg;

struct_summ(isub).pval_r = p_value_pearson;
struct_summ(isub).pval_rho = p_value_spear;
struct_summ(isub).pval_r_neg = p_value_pearson_neg;
struct_summ(isub).pval_rho_neg = p_value_spear_neg;


% Update z_est structure
struct_zest(isub).ID = fileName;
struct_zest(isub).z_est = p_est;

% Update z structure
struct_z(isub).ID = fileName;
struct_z(isub).z =(z-mean(z(:)))./std(z(:));

% Remove empty fields
struct_z = struct_z(~cellfun(@isempty,{struct_z.ID}));
struct_zest = struct_zest(~cellfun(@isempty,{struct_zest.ID}));
struct_summ = struct_summ(~cellfun(@isempty,{struct_summ.ID}));

%% OPTIONAL: Plotting
% Show spatial patterns of each component and highlight info (p-value,
% lambda, corr)

%h(isub) = subplot_tight(6,6,isub, 0.08)
h(isub) = subplot(6,6,isub)
 if p_value_pearson_neg <0.05
    title({[fileName] ; [' - l: ' num2str(lambda_values(end)) '; r: ' num2str(r) '; rho: ' num2str(rho)]; ...
         [' - p-l: ' num2str(p_value_lambda_neg) '; p-r: ' num2str(p_value_pearson_neg) '; p-rho: ' num2str(p_value_spear_neg)]},'Color','r', 'FontSize',7);
     %xlabel(['p: ' num2str(spoc_res(1,2))], 'Color','r')
 else 
    title({[fileName] ; [' - l: ' num2str(lambda_values(end)) '; r: ' num2str(r) '; rho: ' num2str(rho)]; ...
         [' - p-l: ' num2str(p_value_lambda_neg) '; p-r: ' num2str(p_value_pearson_neg) '; p-rho: ' num2str(p_value_spear_neg)]},'FontSize', 7);
     %xlabel(['p: ' num2str(spoc_res(1,2))], 'Color','r')
 end 
 
topoplot(A(:,end),EEG.chanlocs);
colormap('viridis');  %from here: https://github.com/moffat/matlab
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
set(gca,'visible','off')
hold on;

end


% Save current figure
saveas(gcf,[spoc_path cond{folder} '/SA/summaries/SPoC_topo_SA_p_pear_' cond{folder} '.png']);

%% Group-level stats

% Compute a one-sample, one-sided t.test to assess whether the mean
% correlation value between z and z_est is sig. lower than 0 (aka
% negative - coherently with our hypothesis). 

%[stats.h,stats.p,stats.ci,stats.stats] = ttest([struct_summ.r],0,'Tail','left');
[stats.h,stats.p,stats.ci,stats.stats] = ttest([struct_summ.rho],[struct_summ.rho_perm_avg]);
[stats_r.h,stats_r.p,stats_r.ci,stats_r.stats] = ttest([struct_summ.r],[struct_summ.r_perm_avg],'Tail','left');

% Save struct
save([spoc_path cond{folder} '/SA/summaries/SPOC_groupstats_SA_p_'  cond{folder} '.mat'], 'stats_r');

%%  Save the general summary structure
save([spoc_path cond{folder} '/SA/summaries/SPOC_results_SA_p_'  cond{folder} '.mat'], 'SPOC_results');

table_summ = struct2table(struct_summ);
table_z = struct2table(struct_z);
table_zest = struct2table(struct_zest);

save([spoc_path cond{folder} '/SA/summaries/SPOC_results_SA_p_'  cond{folder} '.mat'], 'SPOC_results');  % results summary .mat structure
writetable(table_summ,[spoc_path cond{folder} '/SA/summaries/_summary_SA_p_.csv']);  % lambda, p.values and corr. coeff summary

% Save the 2 tables needed for Results/
writetable(table_z,[results_path cond{folder} '/targetTableSPoC_SA_p_' cond{folder} '.csv'],'WriteVariableNames',0);  % target z
writetable(table_zest,[results_path cond{folder} '/predictionTableSPoC_SA_p_' cond{folder} '.csv'],'WriteVariableNames',0);  % estimated z

%% Clear all the summary structures and tables and close figures
%clear SPOC_results struct_summ table_summ struct_z struct_zest table_z table_zest;
close all;

end 
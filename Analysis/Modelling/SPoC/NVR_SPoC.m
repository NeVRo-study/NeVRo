%% NeVRo_SPoC
%This script computes SPoC between the alpha-band filtered components (8-12 Hz) and
%the continous ratings of the participants (z).

%% Clean previous mess

clc
clear all
%close all

%% Set paths

% Add general matlab scripts and eeglab plugins
addpath(genpath('/Users/Alberto/Documents/MATLAB/'));

% Open eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Paths
master_path = '/Users/Alberto/Documents/PhD/PhD_Side/NeVRo/';
ssd_path = [master_path 'Data/EEG/07_SSD/'];
rat_path = [master_path 'Data/ratings/continuous/not_z_scored/']; 
spoc_path = [master_path 'Data/EEG/08.1_SPOC/']; 
results_path = [master_path 'Results/EEG/SPOC/']; 

% Folders
cond = {'nomov','mov'};

%% Load Data

for fold = 1:length(cond)
     
rawDataFiles = dir([ssd_path cond{fold} '/SBA/narrowband/*.set']);  %we specifcally use SBA data

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
[EEG,com]= pop_loadset([ssd_path cond{fold} '/SBA/narrowband/' loadName]);
EEG.setname=fileName;
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%% Choose the correct ratings and the selected SSD components to be extracted:

% SSD Selected Components file
ssd_tab = readtable([ssd_path cond{fold} '/SSD_selected_components_' cond{fold} '.csv']); %import the selected components table
%ssd_sel =  str2num(cell2num(ssd_tab{str2num(fileName(6:end)),2})); all together
    
sel_sub = ssd_tab(str2num(fileName(6:end)),2); %select the correspondent participant's row
ssd_sel = str2num(cell2mat(table2array(sel_sub))); %convert selected row to list of numbers (indexes indicating components - Terrible sequence of nested functions but it works)

% Break the loop and start with a new participants if selected ssd
% components are less than <4
if (length(ssd_sel)<4) 
    continue 
end 

% Ratings
if exist([rat_path 'nomov/SBA/' fileName '_run_1_alltog_rat_z.txt'], 'file')
    rat_tab=readtable([rat_path 'nomov/SBA/' fileName '_run_1_alltog_rat_z.txt']);
else
    rat_tab=readtable([rat_path 'nomov/SBA/' fileName '_run_2_alltog_rat_z.txt']);
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
epochs=zeros(270,2); %SBA

for i=1:270
    
    epochs(i,1)=i;
    epochs(i,2)=1;
    
end 

% 1. First approach: SPoC via EEGLAB (currently not in use)
% Import events on the EEG file to allow spoc (currently using 1,2,3 -
% Felix's CSP events)
%[EEG, eventnumbers] = pop_importevent(EEG, 'append','no','event',epochs,'fields',{'latency','type'},'skipline',0,'timeunit',1,'align',1,'optimalign','on');

% Run SPOC(need to adapt the epoch size):
%EEG = pop_spoc(EEG,['z.mat'],'500','-0.5 0.496'); %SBA

% 2. Second (actual) approach: "raw" code
%Create 1 second epochs
EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], 'newname', EEG.setname, 'epochinfo', 'yes');

% 2.Original spoc function 
X = permute(EEG.data, [2, 1, 3]);

% Define parameters of X that we will need for further processing
Te = length(X(:,:,1));  % number of samples per epoch
Nx = length(X(1,:,1));  % number of sensors

[W, A, lambda_values, p_values_lambda, Cxx, Cxxz, Cxxe] = spoc(X, z, 'n_bootstrapping_iterations',500);

%%  Extract the best component(s) for each participant (spoc)

%spoc_res = [lambda_values p_values_lambda' A' R'];
spoc_res = [lambda_values p_values_lambda' A', W'];
[spoc_res, index] = sortrows(spoc_res, 1, 'ascend');

%% Compute z_est (the estimated time course of the behavioural data - the univariate variable z - with the best set of weights W)

s_est = zeros(Te, length(z));

for k=1:length(z)
    s_est(:,k) = squeeze(X(:,:,k)) * W(:,index(1));
end

p_est = var(s_est);

%% Compute correlation between z_est and z (in this case there's no training/test separation - might be overfitted)

r_tot = corrcoef(p_est', z');
r = r_tot(1,2:end);

%% Save Spatial Patterns
dlmwrite([spoc_path cond{fold} '/SBA/' fileName '_A.csv'], A); % save spatial patterns
save([spoc_path cond{fold} '/SBA/' fileName '_A.mat'], 'A'); %spatial patterns

%dlmwrite([spoc_path cond{fold} '/SBA/' fileName '_A.csv'],comp_tc); %time course

%% Add everything to the general SPOC_results structure (one per condition - only considering the best SPOC component)

SPOC_results.chanlocs = EEG.chanlocs;
SPOC_results.results(isub).participant = fileName;
SPOC_results.results(isub).preprocessing = EEG.etc;
%SPOC_results.results(isub).SPOC_model.data = X;
SPOC_results.results(isub).SPOC_model.Cov = Cxx ;
SPOC_results.results(isub).SPOC_model.Covz = Cxxz;
SPOC_results.results(isub).SPOC_model.Covtr = Cxxe;
SPOC_results.results(isub).SPOC_model.z_original = z;
SPOC_results.results(isub).SPOC_model.z = (z-mean(z(:)))./std(z(:));
SPOC_results.results(isub).SPOC_model.z_est = p_est;
SPOC_results.results(isub).SPOC_model.corr = r;
SPOC_results.results(isub).SPOC_model.lambda = lambda_values;
SPOC_results.results(isub).SPOC_model.p_values_lambda = p_values_lambda;
SPOC_results.results(isub).stats = p_values_lambda;
SPOC_results.results(isub).weights.SPOC_A = A(:,index(1));
SPOC_results.results(isub).weights.SPOC_W = W(:,index(1));
SPOC_results.results(isub).weights.SSD_W_sel = W_ssd;
SPOC_results.results(isub).weights.SSD_A_sel = A_ssd;

%% Update summary structure

struct_summ(isub).ID = fileName;
struct_summ(isub).lambda = spoc_res(1,1);
struct_summ(isub).pval = spoc_res(1,2);
struct_summ(isub).r = r;

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

%% Group-level stats

% Compute a one-sample, one-sided t.test to assess whether the mean
% correlation value between z and z_est is sig. lower than 0 (aka
% negative - coherently with our hypothesis). 

[stats.h,stats.p,stats.ci,stats.stats] = ttest([struct_summ.r],0,'Tail','left')

% Save struct
save([spoc_path cond{fold} '/SBA/summaries/SPOC_groupstats_'  cond{fold} '.mat'], 'stats');

%% Plotting

%h(isub) = subplot_tight(6,6,isub, 0.06)
h(isub) = subplot(6,6,isub)
 if spoc_res(1,2) <0.05
     title({[fileName] ; ['p: ' num2str(spoc_res(1,2)) ' - l: ' num2str(spoc_res(1,1)) ' - r: ' num2str(r)]},'Color','r');
     %xlabel(['p: ' num2str(spoc_res(1,2))], 'Color','r')
 else 
     title({[fileName] ; ['p: ' num2str(spoc_res(1,2)) ' - l: ' num2str(spoc_res(1,1)) ' - r: ' num2str(r)]});
    % xlabel(['p: ' num2str(spoc_res(1,2))])
 end 
topoplot(A(:,1),EEG.chanlocs);
colormap('viridis');  %from here: https://github.com/moffat/matlab
set(gca,'visible','off')
hold on;

end

% Save the current figure
%sgtitle(['Spatial patterns of the ' cond{fold} ' condition'])
saveas(gcf,[spoc_path cond{fold} '/SBA/summaries/SPoC_topo_' cond{fold} '.png']);

%%  Save the general summary structure
%save([spoc_path cond{fold} '/SBA/summaries/SPOC_results_'  cond{fold} '.mat'], 'SPOC_results');

table_summ = struct2table(struct_summ);
table_z = struct2table(struct_z);
table_zest = struct2table(struct_zest);

save([spoc_path cond{fold} '/SBA/summaries/SPOC_results_'  cond{fold} '.mat'], 'SPOC_results');  % results summary .mat structure
writetable(table_summ,[spoc_path cond{fold} '/SBA/summaries/_summary.csv']);  % lambda, p.values and corr. coeff summary

% Save the 2 tables needed for Results/
writetable(table_z,[results_path cond{fold} '/targetTableSPoC_' cond{fold} '.csv'],'WriteVariableNames',0);  % target z
writetable(table_zest,[results_path cond{fold} '/predictionTableSPoC_' cond{fold} '.csv'],'WriteVariableNames',0);  % estimated z

%% Clear all the summary structures and tables and close figures
clear SPOC_results struct_summ table_summ struct_z struct_zest table_z table_zest;
close all;

end 
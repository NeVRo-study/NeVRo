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

% Folders
cond = {'nomov','mov'};

%% Load Data

for fold = 1:length(cond)
    
    
rawDataFiles = dir([ssd_path cond{fold} '/SBA/narrowband/*.set']);  %we specifcally use SBA data

%NB: for the mov condition, S09, S24 and S27 are problematic because none of
%their SSD components passed our selection criteria 

% Initialize all the SPoC summary tables
%SPOC_Table_lambda= ones(length(rawDataFiles),2);
spoc_t = ones(length(rawDataFiles),3);
SPOC_Table_A= ones(30,length(rawDataFiles)+1);
SPOC_Table_W= ones(30,length(rawDataFiles)+1);
%SPOC_Table_TC=ones(67502,24);

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

% Import events on the EEG file to allow spoc (currently using 1,2,3 -
% Felix's CSP events)
%[EEG, eventnumbers] = pop_importevent(EEG, 'append','no','event',epochs,'fields',{'latency','type'},'skipline',0,'timeunit',1,'align',1,'optimalign','on');

% Create 1 second epochs
EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], 'newname', EEG.setname, 'epochinfo', 'yes');

% Run SPOC:
% 1.Via EEGLAB function (need to adapt the epoch size 
%EEG = pop_spoc(EEG,['z.mat'],'500','-0.5 0.496'); %SBA

% 2.Original spoc function 
X = permute(EEG.data, [2, 1, 3]);
[W, A, lambda_values, p_values_lambda, Cxx, Cxxz, Cxxe] = spoc(X, z, 'n_bootstrapping_iterations',500);

%% Extract the best component(s) for each participant (pop_spoc)  - double check the location of the final A,W matrices in the EEGlab structure

%If you use pop_spoc:
%First, let's sort the resulting SPoC structure in every single field
% Sfields = fieldnames(EEG.dipfit.model);
% Scell = struct2cell(EEG.dipfit.model);
% sz = size(Scell);
% 
% % Convert to a matrix
% Scell = reshape(Scell, sz(1), []);      % Px(MxN)
% % Make each field a column
% Scell = Scell';                         % (MxN)xP
% % Sort by second field "p.value"
% Scell = sortrows(Scell, 2);
% 
% % Save lambdas and p.value
% spoc_t(isub,1) = str2num(fileName(6:7));
% spoc_t(isub,2) = Scell{1,1}; %lambda
% spoc_t(isub,3) = Scell{1,2}; %p.value
% 
% % Extract the z-scored behavioural target variable
% target_z = zscore(EEG.SPoC_z');
% 
% % Save the respective A matrix (EEG.icawinv) and put it in the general table
% A=EEG.icawinv;
% dlmwrite([spoc_path cond{fold} '/' fileName '_A.csv'],A);  %spatial patterns
% % Save also the best time course and put i
% comp_tc=EEG.icaact;
% dlmwrite([spoc_path cond{fold} '/' fileName '_A.csv'],comp_tc); %time course

%%  Extract the best component(s) for each participant (spoc)

spoc_res = [lambda_values p_values_lambda' A'];
spoc_res = sortrows(spoc_res, 1, 'ascend');

spoc_t(isub,1) = str2num(fileName(6:7));
spoc_t(isub,2) = spoc_res(1,1); %lambda
spoc_t(isub,3) = spoc_res(1,2); %p.value

dlmwrite([spoc_path cond{fold} '/SBA/' fileName '_A.csv'], A);
save([spoc_path cond{fold} '/SBA/' fileName '_A.mat'], 'A'); %spatial patterns
save([spoc_path cond{fold} '/SBA/' fileName '_spoc_res.mat'], 'spoc_res'); 
%dlmwrite([spoc_path cond{fold} '/SBA/' fileName '_A.csv'],comp_tc); %time course

%% Plotting

%h(isub) = subplot_tight(6,6,isub, 0.06)
h(isub) = subplot(6,6,isub)
title({[fileName] ; ['p: ' num2str(spoc_res(1,2)) ' - l: ' num2str(spoc_res(1,1))]});
% if spoc_res(1,2) <0.05
%     xlabel(['p: ' num2str(spoc_res(1,2))], 'Color','r')
% else 
%     xlabel(['p: ' num2str(spoc_res(1,2))])
% end 
topoplot(A(:,1),EEG.chanlocs);
set(gca,'visible','off')
hold on;
%sgtitle(['Topographies (spatial patterns) of the ' cond{fold} ' condition'])
%saveas(gcf,[spoc_path 'SPoC_topo_' cond{fold} '.png']);

end

csvwrite([spoc_path 'SPoC_pvals_' cond{fold} '.csv'],spoc_t);  %time course
%sgtitle(['Topographies (spatial patterns) of the ' cond{fold} ' condition'])
saveas(gcf,[spoc_path 'SPoC_topo_lambda_' cond{fold} '.png']);
close all;

end 
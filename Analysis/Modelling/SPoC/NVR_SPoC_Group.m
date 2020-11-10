%% NeVRo_SPoC_Group

% This script allows to compute and plot SpoC's group results:

% 1.Compute the absolute values of each spatial pattern matrix
% 2.Average each absolute A
% 3.Plot the comprehensive average plots 

%% Clean previous mess

clc
clear all
%close all

%% Set paths
% NB: Add your general matlab and eeglab paths

% Open EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Paths
master_path = '../../../../NeVRo/';
addpath(master_path);

ssd_path = [master_path 'Data/EEG/07_SSD/'];
rat_path = [master_path 'Data/ratings/continuous/not_z_scored/']; 
spoc_path = [master_path 'Data/EEG/08.1_SPOC/']; 

% Folders
cond = {'nomov','mov'};

% Load chanlocs
load('chans.mat');

% IF you need/want: fix radius to fit everything "within the scalp"
for n=1:length(chans)
       chans(1,n).radius = chans(1,n).radius*0.795;
end

%% Load Data

for folder = 1:length(cond)
       
rawDataFiles = dir([spoc_path cond{folder} '/SBA/*A.mat']);  %we specifcally use SBA data

A_group = ones(32,length(rawDataFiles));

for isub = 1:length(rawDataFiles)
    
loadName = rawDataFiles(isub).name;
fileName = loadName(1:7);

% Load SPoC's results
load([rawDataFiles(isub).folder '/' rawDataFiles(isub).name]);

% Select the A matrix of the best SPoC component
% A = spoc_res(1,3:end)'; %invert it into col form

A = A(:,end);

% A/magnitude(A) aka:
A_mag = abs(A/norm(A));

%..then average of all the single subject A
A_group(:,isub) = abs(A);
A_group_mag(:,isub) = A_mag;

end 

%A_avg_abs = mean(A_group_abs,2); %mean with absolute values
A_avg = mean(A_group,2); %mean with all values
A_avg_mag = mean(A_group_mag,2);


%%  Plot

% Average 
figure('units','normalized','outerposition',[0 0 1 1]);
subplot(1,2,1);
topoplot(A_avg,chans);
colormap('viridis');
h = colorbar;
set(h,'Yticklabel',linspace(min(A_avg),max(A_avg)));
title(['Average Spatial Patterns - ' cond{folder}]);

% Average magnitude
subplot(1,2,2);
topoplot(A_avg_mag,chans);
colormap('viridis');
j = colorbar;
set(j,'Yticklabel',linspace(min(A_avg_mag),max(A_avg_mag)));
title(['Average Spatial Patterns (with magnitude ratio) - ' cond{folder}])

saveas(gcf,['avg_topoplots_' cond{folder} '.jpeg']);
close all;

end 

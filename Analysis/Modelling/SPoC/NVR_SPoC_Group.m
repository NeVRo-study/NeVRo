%% NeVRo_SPoC_Group
% This script allow to compute the group results of SPoC:
% 1.Compute the absolute values of each spatial pattern matrix
% 2.Average each absolute A
% 3.Plot the comprehensive average plots 

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

% Load chanlocs
load('chans.mat');

% IF you need/want: Fix radius to fit everything "within the scalp"
for n=1:length(chans)
       chans(1,n).radius = chans(1,n).radius*0.795;
end

%% Load Data

for fold = 1:length(cond)
       
rawDataFiles = dir([spoc_path cond{fold} '/SBA/*res.mat']);  %we specifcally use SBA data

A_group = ones(32,length(rawDataFiles));

for isub = 1:length(rawDataFiles)
    
loadName = rawDataFiles(isub).name;
fileName = loadName(1:7);

% Load spoc results
load([rawDataFiles(isub).folder '/' rawDataFiles(isub).name]);

% Select the A matrix of the best SPoC component
A = spoc_res(1,3:end)'; %invert it into col form
%A_abs = abs(A); %is it correct? NO NEED TO DO IT WITH THE MAGNITUDE

%A/magnitude(A) aka:
A_mag = A/norm(A);
%then Average of all the single subject A
%plot it!

A_group(:,isub) = A;
%A_group_abs(:,isub) = A_abs;

A_group_mag(:,isub) = A_mag;
end 

%A_avg_abs = mean(A_group_abs,2); %mean with absolute values
A_avg = mean(A_group,2); %mean with all values
A_avg_mag = mean(A_group_mag,2);
end 


%% Let's plot them
% Plot
figure;
subplot(1,2,1);
topoplot(A_avg,chans);
title({'Average Spatial Patterns'})
colorbar;

subplot(1,2,2);
topoplot(A_avg_mag,chans);
title({'Average Spatial Patterns (with magnitude ratio)'})
colorbar;

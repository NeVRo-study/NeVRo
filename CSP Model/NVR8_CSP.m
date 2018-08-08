%% NVR CSP
% 2017 by Felix Klotzsche* and Alberto Mariola
% *: main contribution
%
%This script applies CSP to the data and saves some performance measures.
%

function NVR8_CSP(cropstyle, ssd_freq, keep_filt)

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:

if keep_filt
    filt_folder = 'SSDfiltered';
else
    filt_folder = 'unfiltered';
end
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/' cropstyle '/3_SSD/' filt_folder '/' ssd_freq '/'];
path_results = [path_dataeeg 'F_EEG/' cropstyle '/4_CSP/' filt_folder '/' ssd_freq '/'];
path_reports = [path_results 'reports/'];
if ~exist(path_reports, 'dir'); mkdir(path_reports); end

%% === CONSTANTS ===
freqs = [6 8 13 15];
timewnd = [-0.5 0.5]; 
mrks = {'1','3'};

trainsets = {[path_in_eeg '*.set']};

run('E:\Felix\BCILAB\BCILAB-devel\bcilab.m')

% === DEFINE APPROACHES ===
% (note: use unique names for all your approaches)

approach = {'CSP' 'SignalProcessing', {'EpochExtraction', timewnd, ...
                                       'Resampling', 0, ...
                                       'FIRFilter', 0}, ...
    'Prediction', {'FeatureExtraction', {'PatternPairs', 3, ...
                                         'LogTransform', 1, ...
                                         'ShrinkageCovariance', 1},...
                   'MachineLearning', {'Learner', {'lda', ...
                                                   'Regularizer', 'shrinkage', ...
                                                   'Lambda', search([0:0.2:1])}}}};

% === RUN BATCH ANALYSIS ===

results = bci_batchtrain('StudyTag','CSPBatch', ...
    'Data',trainsets, ...
    'Approaches',approach, ...
    'TargetMarkers',mrks, ...
    'ReuseExisting',false, ...
    'TrainArguments',{'EvaluationScheme',[10 3], ... 
                      'OptimizationScheme', 5}, ... 
    'StoragePattern',[path_results, '%approach-%set.mat']);



   
    
    

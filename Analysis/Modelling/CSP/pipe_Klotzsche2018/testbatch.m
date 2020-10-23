%% NVR CSP
% 2017 by Felix Klotzsche* and Alberto Mariola
% *: main contribution
%
%This script applies CSP to the data and saves some performance measures.
%
% 


%% === CONSTANTS ===
freqs = [6 8 13 15];
timewnd = [-0.5 0.5]; 
mrks = {'1','3'};
trainsets = {'G:\NEVRO\NVR_master\data/EEG\F_EEG\SBA_SSD\*.set'};
path_results = 'G:\NEVRO\NVR_master\data\EEG\F_EEG\CSP_results\';


% === DEFINE APPROACHES ===
% (note: use unique names for all your approaches)

approach = {'CSP', 'SignalProcessing', {'EpochExtraction', timewnd, ...
                                       'FIRFilter', freqs}, ...
    'Prediction', {'FeatureExtraction', {'PatternPairs', 3, ...
                                         'LogTransform', 1, ...
                                         'ShrinkageCovariance', 1}}
    'MachineLearning', {'Learner','lda', ...
                        'Regularizer', 'auto'}};

% === RUN BATCH ANALYSIS ===

results = bci_batchtrain('StudyTag','CSPBatch', ...
    'Data',trainsets, ...
    'Approaches',approach, ...
    'TargetMarkers',mrks, ...
    'ReuseExisting',false, ...
    'TrainArguments',{'EvaluationScheme',{'chron',10,1}}, ... 
    'StoragePattern',[path_results, '%approach-%set.mat']);




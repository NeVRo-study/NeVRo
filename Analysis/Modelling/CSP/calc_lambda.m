

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/CSP_results2/']; % [path_dataeeg 'SBA_PREP/']; %



%1.2 Get data files
files_eeg = dir([path_in_eeg '*.mat']);
files_eeg = {files_eeg.name};

lambdas = [];

for fifi = 1:length(files_eeg)
    
    load(strcat(path_in_eeg, files_eeg{fifi}), 'res');
    cur_la = res.model.args.pred.ml.learner.plambda;
    
    lambdas = [lambdas cur_la];
    clear('res')
end
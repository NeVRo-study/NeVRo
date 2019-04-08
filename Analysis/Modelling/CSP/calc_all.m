
%%
%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/SBA/4_CSP/SSDfiltered/6/']; % [path_dataeeg 'SBA_PREP/']; %

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.mat']);
files_eeg = {files_eeg.name};

lambdas = [];
pats = [];
losses = [];

for fifi = 1:length(files_eeg)
    
    load(strcat(path_in_eeg, files_eeg{fifi}), 'res');
    cur_la = res.model.args.pred.ml.learner.plambda;
    cur_pa = res.model.featuremodel.patterns(6,:);
    cur_lo = res.loss;
    lambdas = [lambdas cur_la];
    pats = [pats; cur_pa];
    losses = [losses cur_lo];
    clear('res')
end
%%
mean_pats = mean(pats, 1);
figure(3)
topoplot(mean_pats, EEG.chanlocs)
%% NVR_EventsAro
%This script adds the arousal events to the NeVRo EEG data. 

function NVR_05_ICA(cropstyle, mov_cond) 

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
path_data = '../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg 'eventsAro/' mov_cond '/' cropstyle '/']; 

% output paths:
path_out_eeg = [path_dataeeg 'ICA/' mov_cond '/' cropstyle '/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

%1.3 Launch EEGLAB:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

for isub = 1:length(files_eeg)
    %for itask=1:length(task_red)
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = strcat(thissubject, mov_cond, '_PREP_', cropstyle, '_eventsaro'); 
    filename = char(filename);
    
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);

    eeg_rank = rank(EEG.data);
    
    EEG = pop_runica(EEG, 'extended',1,'interupt','on','pca',eeg_rank);
    
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset(EEG, [filename  '_ICA.set'] , path_out_eeg);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        
end
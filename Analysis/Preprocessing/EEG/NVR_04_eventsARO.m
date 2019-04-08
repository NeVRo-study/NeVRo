%% NVR_EventsAro
% This script adds the arousal events to the NeVRo EEG data. 
% Requires behavioral arousal rating data in seperate folder (ratings/...).
%
% 2017 (edited 2018) by Felix Klotzsche 

function NVR_04_eventsARO(cropstyle, mov_cond) 

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
path_data = '../../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg '03_CROP/' mov_cond '/' cropstyle '/']; 
path_evaro_rat = [path_data 'ratings/class_bins/' mov_cond '/' cropstyle '/' ];

% output paths:
path_out_eeg = [path_dataeeg '04_eventsAro/' mov_cond '/' cropstyle '/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};
files_evaro = dir([path_evaro_rat '*.txt']);
files_evaro = {files_evaro.name};


%1.3 Launch EEGLAB:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

for isub = 1:length(files_eeg)
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = strcat(thissubject, mov_cond, '_PREP_', cropstyle); 
    filename = char(filename);
    
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    %% 3.Import event info
    % get the according event file
    whos_events = thissubject; %'dummy'; % 
    f_index = strncmp(whos_events, files_evaro, length(whos_events)); %thissubject
    try
        file_evaro = files_evaro{f_index};
    catch
        fprintf(['No event file for ', thissubject, '. Skipped.']);
        continue;
    end
        
    % load the events into the EEGSET
    % - overwriting old events ('append', 'no')
    EEG = pop_importevent( EEG, ...
        'append','no', ...
        'event',[path_evaro_rat file_evaro], ...
        'fields',{'latency' 'type'}, ...
        'skipline',1, ...
        'timeunit',0.001, ...
        'optimalign','off');
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset(EEG, [filename  '_eventsaro.set'] , path_out_eeg);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        
end
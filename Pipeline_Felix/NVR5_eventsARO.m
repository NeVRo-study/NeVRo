%% NVR_EventsAro
%This script adds the arousal events to the NeVRo EEG data. 

function NVR5_eventsARO(cropstyle) 

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_raweeg = [path_dataeeg 'F_EEG/' cropstyle '/0_cropped/']; % [path_dataeeg 'SBA_PREP/']; %

if strcmp(cropstyle, 'SBA')  
        path_evaro_rat = [path_master 'data/ratings/preprocessed/' ...
            'z_scored_alltog/alltog/nomove/1Hz/aro_epochs/'];
elseif strcmp(cropstyle, 'SA')
        path_evaro_rat = [path_master 'data/ratings/preprocessed/' ...
            'z_scored_comb/comb/nomove/1Hz/aro_epochs/'];
end



% output paths:
path_evaro_eeg = [path_dataeeg 'F_EEG/' cropstyle '/1_eventsAro/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_evaro_eeg, 'dir'); mkdir(path_evaro_eeg); end


%1.2 Get data files
files_eeg = dir([path_raweeg '*.set']);
files_eeg = {files_eeg.name};
files_evaro = dir([path_evaro_rat '*.txt']);
files_evaro = {files_evaro.name};


%1.3 Launch EEGLAB:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

for isub = 1:length(files_eeg)
    %for itask=1:length(task_red)
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, '_PREP');
    thissubject = thissubject{1};
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = strcat(thissubject, '_PREP_', cropstyle); % _PREP_SBA
    filename = char(filename);
    
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_raweeg, filename '.set']);
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
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
    EEG = pop_saveset(EEG, [filename  '_eventsaro.set'] , path_evaro_eeg);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
        
end
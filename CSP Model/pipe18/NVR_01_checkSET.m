%% NVR checkSET
% 2017 by Felix Klotzsche 
% 
% This script loads the raw and continuous EEG data for either MOV or NOMOV 
% condition and performs a sanity check:
% Are any weird markers included (esp. boundary events)? 7 events are
% expected (1 boundary at the very start + 2 events (start+end) per roller 
% coaster and the break)
%
% Data directory (path_in_eeg) has to be defined manually to read in the 
% according files . 

%% 1.Set Variables
clc
%clear all

%1.1 Set different paths:
% path to NeVRo/Data:
path_master = 'D:/Felix/ownCloud/NeVRo/Data/';
% input paths:
path_dataeeg = [path_master 'EEG/raw/'];
path_in_eeg = [path_dataeeg 'nomov_SETs/']; %[path_dataeeg 'mov_SETs']


%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

%1.3 Launch EEGLAB:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

n_eve = zeros(length(files_eeg),1);
win_length = zeros(length(files_eeg),1);

for isub = 5:length(files_eeg)
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, '.');
    thissubject = thissubject{1};
    
    % Uncomment the following for debugging with single subject:
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = files_eeg{isub};
    filename = strsplit(filename, '.');
    filename = filename{1};
    
    
    
    %% 2.Load EEG data
    
    [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
    EEG = eegh(com,EEG);
    
    
    % Check number of events:       
    
    n_eve(isub) = size(EEG.event, 2);
    win_length(isub) = EEG.xmax;
    
    if (~(size(EEG.event, 2) == 7))
        fprintf('###############\n\n\n\n\n\n\n\n###############');
        fprintf(['Problem with subject ' thissubject '.\n']);
        fprintf(['There were ' num2str(size(EEG.event, 2)) ' events.\n']);
        fprintf('###############\n\n\n\n\n\n\n\n###############');
        break; %continue
    end
    
    fprintf('###############\n');
    fprintf(['No problem with subject ' thissubject '.\n']);
    fprintf([num2str(size(EEG.event, 2)) ' events found.\n']);
    fprintf('###############\n');
      
    
end


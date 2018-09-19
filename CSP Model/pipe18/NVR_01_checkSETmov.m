%% NVR checkSET
% 2017 by Felix Klotzsche 
% *: main contribution

% This script loads the raw and continuous EEG data (saved as EEGLAB SET 
% files) and performs a sanity check:
% 0: 25 events included? (24 exp triggers + 1 boundary event)

%% 1.Set Variables
clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'H:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'EEG_raw_cont/mov_SETs/']; 

% output paths:
path_out_eeg = [path_dataeeg 'EEG_raw_cont/mov_SETs']; 
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end


%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

%1.3 Launch EEGLAB:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

n_eve = zeros(length(files_eeg),1);
win_length = zeros(length(files_eeg),1);

for isub = 1:length(files_eeg)
    
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
        continue
    end
    
    fprintf('###############\n');
    fprintf(['No problem with subject ' thissubject '.\n']);
    fprintf([num2str(size(EEG.event, 2)) ' events found.\n']);
    fprintf('###############\n');
    
    
    %% 2.Import EEG data as .set files
    %         [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    %         EEG = eegh(com,EEG);
    %         EEG.setname=filename;
    %         [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    %         eeglab redraw;
   
    
end


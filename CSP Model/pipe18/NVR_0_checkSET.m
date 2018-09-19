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
path_in_eeg = [path_dataeeg 'EEG_raw_cont/oSETs/']; 

% output paths:
path_out_eeg = [path_dataeeg 'EEG_raw_cont/mov_SETs']; 
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end


%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

%1.3 Launch EEGLAB:
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

for isub = 37:length(files_eeg)
    
    
    
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
    
    
    % Cut out continuous mov and no-mov segments:
    
    % Relevant events (watch out for retarded spacing): 
    % S 30	Space Movement Start
    % S 35	Ande Movement End
    % S130	Space No Movement Start
    % S135	Ande No Movement End
    
    % find timing of markers
    mov_mrkrs = {'S 30' 'S 35'};
    nomov_mrkrs = {'S130' 'S135'};
    [ idx_mov_start] = find(strcmp({EEG.event.type}, mov_mrkrs{1}));
    [ idx_mov_end] = find(strcmp({EEG.event.type}, mov_mrkrs{2}));
    [ idx_nomov_start] = find(strcmp({EEG.event.type}, nomov_mrkrs{1}));
    [ idx_nomov_end] = find(strcmp({EEG.event.type}, nomov_mrkrs{2}));
    
    lat_mov_start = EEG.event(idx_mov_start).latency;
    lat_mov_end = EEG.event(idx_mov_end).latency;
    lat_nomov_start = EEG.event(idx_nomov_start).latency;
    lat_nomov_end = EEG.event(idx_nomov_end).latency;
    
    EEG = pop_select( EEG,'point',[lat_mov_start lat_mov_end] );
    
    
    
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
    
    EEG.setname=thissubject;
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset( EEG, [thissubject] , path_out_eeg);
    eeglab redraw;
    
    
    
    %% 2.Import EEG data as .set files
    %         [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    %         EEG = eegh(com,EEG);
    %         EEG.setname=filename;
    %         [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    %         eeglab redraw;
   
    
end


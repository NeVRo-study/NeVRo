%% NVR import
% 2017 by Felix Klotzsche and Alberto Mariola*
% *: main contribution

%This script loads the raw EEG data to EEGLAB and sets it up for further
%processing

%% 1.Set Variables
clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'EEG_raw_cont/']; % [path_dataeeg 'SBA_PREP/']; %

% output paths:
path_out_eeg = [path_dataeeg 'EEG_raw_cont/SETs']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end


%1.2 Get data files
files_eeg = dir([path_in_eeg '*.vhdr']);
files_eeg = {files_eeg.name};

%1.3 Launch EEGLAB:
run('E:\Felix\BCILAB\BCILAB-devel\dependencies\eeglab13_4_4b\eeglab.m');
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

for isub = 1:length(files_eeg)
    
    
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, '_Seg');
    thissubject = thissubject{1};
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = files_eeg{isub};
    filename = strsplit(filename, '.');
    filename = filename{1};
    
    
    
    %% 2.Import EEG data
    
    [EEG, com] = pop_loadbv(path_in_eeg, strcat(filename,'.vhdr'));
    EEG = pop_chanedit(EEG, 'lookup','E:\Felix\BCILAB\BCILAB-devel\dependencies\eeglab13_4_4b/plugins/dipfit2.3/standard_BEM/elec/standard_1005.elc', ...
        'eval','chans = pop_chancenter( chans, [],[]);');
    EEG = eegh(com,EEG);
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


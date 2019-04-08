%% NVR crop and combine
% 2017 by Felix Klotzsche and Alberto Mariola*
% *: main contribution
%
%This script crops out the significant parts of the EEG data stream (SPACE
%coaster, BREAK, ANDES coaster) and combines them to a single stream.
%From the coaster parts, 2.5 secs at the beginning and at the end are cut
%off.

function NVR4_crop_SBA(cropstyle)
%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/SBA_EOGreg/']; % [path_dataeeg 'SBA_PREP/']; %

% output paths:

path_out_eeg = [path_dataeeg 'F_EEG/' cropstyle '/0_cropped/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};


%1.3 Launch BCILAB & EEGLAB:
% cur_path = pwd;
% cd('E:\Felix\BCILAB\BCILAB-devel\');
% bcilab;
% cd(cur_path)

run('E:\Felix\BCILAB\BCILAB-devel\dependencies\eeglab13_4_4b\eeglab.m');



for isub = 1:length(files_eeg)
    
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    
    %1.5 Set filename:
    filename = files_eeg{isub};
    filename = strsplit(filename, '.');
    filename = filename{1};
    
    % Get data:
    [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
    
    % Cut out Space:
    EEG = pop_rmdat( EEG, {'S130'},[2.5 150.5] ,0);
    EEG.setname = 'Space';
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    
    
    if strcmp(cropstyle, 'SBA')    
        % Get data:
        [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
        % Cut out Break:
        EEG = pop_rmdat( EEG, {'S132'},[0 30] ,0);
        EEG.setname = 'Break';
        [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);    
    end
    
    % Get data:
    [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
    % Cut out Break:
    EEG = pop_rmdat( EEG, {'S134'},[2.5 94.5] ,0);
    EEG.setname = 'Andes';
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    
    % Merge them:
    if strcmp(cropstyle, 'SBA')  
        parts = [1  2  3];
    elseif strcmp(cropstyle, 'SA')
        parts = [1 2];
    end
            
    EEG = pop_mergeset(ALLEEG, parts, 0);
    EEG.setname = [filename '_' cropstyle];
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset( EEG, [EEG.setname  '.set'] , path_out_eeg);
end

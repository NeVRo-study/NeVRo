%% NVR crop and combine
% 2017 by Felix Klotzsche and Alberto Mariola*
% *: main contribution
%
%This script crops out the significant parts of the EEG data stream (SPACE
%coaster, BREAK, ANDES coaster) and combines them to a single stream.
%From the coaster parts, 2.5 secs at the beginning and at the end are cut
%off.

function NVR_03_crop(cropstyle, mov_cond)
%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% input paths:
path_dataeeg =  '../../Data/EEG/';
path_in_eeg = [path_dataeeg 'PREP3/' mov_cond '/']; 

% output paths:

path_out_eeg = [path_dataeeg 'CROP/' cropstyle '/']; 
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

% Define relevant events (watch out for retarded spacing): 
    % S 30	Space Movement Start
    % S 32  Break Movement Start
    % S 34	Ande Movement Start
    % S130	Space No Movement Start
    % S132  Break Movement Start
    % S134	Ande No Movement Start
    
mov_mrkrs = {'S 30' 'S 32' 'S 34'};
nomov_mrkrs = {'S130' 'S132' 'S134'};

switch mov_cond
    case 'mov'
        mrkrs = mov_mrkrs;
    case 'nomov' 
        mrkrs = nomov_mrkrs;
end

% How much to trim from beginning and end of each roller coaster (in s)?
trim_s = 2.5;

% lengths of roller coasters (in s):
len_space = 153;
len_break = 30;
len_andes = 97;

for isub = 1:length(files_eeg)
    
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    
    %1.5 Set filename:
    filename = files_eeg{isub};
    filename = strsplit(filename, '.');
    filename = filename{1};
    
    % Get data:
    [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    full_eeg_idx = CURRENTSET; 
    
    % Cut out Space:
    EEG = pop_rmdat( EEG, mrkrs(1),[trim_s (len_space-trim_s)] ,0);
    EEG.setname = 'Space';
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    space_idx = CURRENTSET;
       
    % Get data:
    EEG = ALLEEG(full_eeg_idx);
    % Cut out Break:
    EEG = pop_rmdat( EEG, mrkrs(2),[0 len_break] ,0);
    EEG.setname = 'Break';
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    break_idx = CURRENTSET;
    
    % Get data:
    EEG = ALLEEG(full_eeg_idx);
    % Cut out Andes:
    EEG = pop_rmdat( EEG, mrkrs(3),[trim_s (len_andes-trim_s)] ,0);
    EEG.setname = 'Andes';
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    andes_idx = CURRENTSET;
    
    % Merge them:
    if strcmp(cropstyle, 'SBA')  
        parts = [space_idx break_idx andes_idx];
    elseif strcmp(cropstyle, 'SA')
        parts = [space_idx andes_idx];
    end
            
    EEG = pop_mergeset(ALLEEG, parts, 0);
    EEG.setname = [filename '_' cropstyle];
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset( EEG, [EEG.setname  '.set'] , path_out_eeg);
end

%% NVR SSD
% 2017 by Felix Klotzsche* and Alberto Mariola
% *: main contribution
%
%This script applies SSD to the data and returns the 15 best components.
%
% Nikulin, Nolte, & Curio (2011): A novel method for reliable and fast
% extraction of neural EEG/MEG oscillations on the basis of spatio-spectral
% decomposition, NeuroImage, 55, 1528-1535


function NVR7_SSD(cropstyle, ssd_freq, keep_filt)

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/' cropstyle '/2_artrej/']; % [path_dataeeg 'SBA_PREP/']; %

% output paths:
if keep_filt
    filt_folder = 'SSDfiltered';
else
    filt_folder = 'unfiltered';
end
path_out_eeg = [path_dataeeg 'F_EEG/' cropstyle '/3_SSD/' filt_folder '/' ssd_freq '/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end
path_reports = [path_out_eeg 'reports/'];
if ~exist(path_reports, 'dir'); mkdir(path_reports); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};


%1.3 Launch BCILAB & EEGLAB:
% cur_path = pwd;
% cd('E:\Felix\BCILAB\BCILAB-devel\');
% bcilab;
% cd(cur_path)

run('E:\Felix\BCILAB\BCILAB-devel\dependencies\eeglab13_4_4b\eeglab.m');
[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

for isub = 1:length(files_eeg)
    
    %1.5 Set filename:
    filename = files_eeg{isub};
    filename = strsplit(filename, '.');
    filename = filename{1};
    
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
    %% 2. SSD
    [ALLEEG EEG]=pop_ssd_AM(ALLEEG,EEG,CURRENTSET,...
        ssd_freq, ... %central frequency
        '2', ... % filter order
        1, ... % dimension reduction (y/n)?
        '15', ... % how many components/dimensions?
        keep_filt, ... , % overwrite original data (y/n)?
        0, ... % run on all opened data sets?
        0, ... %use events to epoch?
        '', ... % event names
        []); % otherwise: epoch limits
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
    
    %% 5. Save SSD_set
    EEG = pop_saveset( EEG, [filename  '_SSD.set'] , path_out_eeg);
    %EEG = pop_saveset( EEG, [filename  '_SSD.set'] , ssdpath);
    
end
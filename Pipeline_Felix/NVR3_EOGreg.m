% NVR Regress out EOG activity
% 2017 by Felix Klotzsche
% inspired by Parra (2004)
%
% References:
% Parra L C, Spence C D, Gerson A D and Sajda P (2005):
% Recipes for the linear analysis of EEG
% NeuroImage 28 326–41
%
% Code from:
% http://www.parralab.org/teaching/eeg/logist.m


%This script loads the EEG data after it had been processed by the PREP
%pipeline and uses Parra's regression alogrithm to get rid of EOG activity

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/PREP_full/']; % [path_dataeeg 'SBA_PREP/']; %

% output paths:

path_out_eeg = [path_dataeeg 'F_EEG/regEOG/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

%1.3 List EOG channels:
%eogchannels = [1, 5, 27, 32];
eogchannel_names = {'VEOG', 'HEOG', 'Fp1', 'Fp2'};

%1.3 Launch BCILAB & EEGLAB:
% cur_path = pwd;
% cd('E:\Felix\BCILAB\BCILAB-devel\');
% bcilab;
% cd(cur_path)

run('E:\Felix\BCILAB\BCILAB-devel\dependencies\eeglab13_4_4b\eeglab.m');

[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

%%

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




%% 3. Regress out the EOG channels (code from Parra (2005))
    
    [wa, eogchannels, ste]  = intersect({EEG.chanlocs.labels}, eogchannel_names); 
    data = transpose(EEG.data);
    data = data - data(:,eogchannels) * (data(:,eogchannels)\data); 
    EEG.data = transpose(data);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
       
    % remove EOG channels:
    EEG = pop_select( EEG,'nochannel',{'HEOG' 'VEOG' 'Fp1' 'Fp2'});
    EEG.setname=[filename '_regEOG'];
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset( EEG, [filename  '_regEOG.set'] , path_out_eeg);
    eeglab redraw;
end
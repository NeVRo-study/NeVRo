
%% NVR artifact rejection
% 2017 by Felix Klotzsche*
% *: main contribution
%
%This script calculates which epoch shall be discarded due to high variance
%in the signal. The procedure is oriented at the approach by Haufe, Daehne &
%Nikulin (2014).
%
% Haufe, Dähne, & Nikulin (2014): Dimensionality reduction for the analysis
% of brain oscillations, NeuroImage, 101, 583–97

function NVR6_artrej(cropstyle)

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'F_EEG/' cropstyle '/1_eventsAro/']; % [path_dataeeg 'SBA_PREP/']; %

% output paths:
path_out_eeg_epo = [path_dataeeg 'F_EEG/' cropstyle '/2a_epo/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_out_eeg_epo, 'dir'); mkdir(path_out_eeg_epo); end
path_out_eeg = [path_dataeeg 'F_EEG/' cropstyle '/2_artrej/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
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

% Create report file:
fid = fopen([path_reports 'rejected_epos.csv'], 'a') ;
fprintf(fid, 'ID,n_epos_rejected,epos_rejected\n') ;
fclose(fid);


discarded = {};
discarded_mat = zeros(length(files_eeg),20);
counter = 0;
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
    
    
    %% Epoch it:
    EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], ...
        'newname', [filename '_epo'], ...
        'epochinfo', 'yes');
    EEG = eegh(com,EEG);
    EEG.setname=[filename '_epo'];
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset( EEG, [filename  '_crop_epo.set'] , path_out_eeg_epo);
    eeglab redraw;
    
    %% Bandpass filter (5-45Hz):
    EEG = pop_eegfiltnew(EEG, [], 5, 414, true, [], 0);
    EEG = pop_eegfiltnew(EEG, [], 45, 74, 0, [], 0);
    EEG.setname=[filename '_epo_filt5-45'];
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    %% Calculate variance threshhold:
    EEGsize = size(EEG.data);
    n_chan = EEGsize(1);
    n_trials = EEGsize(3);
    if (n_chan ~= 28 || n_trials ~= 270)
        fprintf('\n\n\nWARNING!!! \nTRIALS:')
        disp(n_trials)
        disp('CHANNELS:')
        disp(n_chan)
    end
    tri_var = ones(n_chan,n_trials);
    for tri = 1:n_trials
        for cha = 1:n_chan
            tri_var(cha,tri) = var(EEG.data(cha,:,tri));
        end
    end
    
    avg_per_trial = mean(tri_var,1);
    q5 = prctile(avg_per_trial,5);
    q95 = prctile(avg_per_trial,95);
    thresh = q95 + 1 * (q95 - q5);
    disc_epo = find(avg_per_trial > thresh);
    disp(filename)
    fprintf('\nTo be discarded:\n');
    disp(disc_epo)
    
    [ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;
    [EEG, com] = pop_loadset([path_in_eeg, filename, '.set']);
    
    rm = [];
    for dis = disc_epo
        rm = [rm; dis-1 dis];
    end
    EEG = pop_select( EEG,'notime',rm );
    EEG.setname=[filename '_artrej'];
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset( EEG, [filename  '_crop_artrej.set'] , path_out_eeg);
    eeglab redraw;
    counter = counter+1;
    discarded_mat(counter,1:length(disc_epo)) = disc_epo;
    discarded{counter} = disc_epo;
    
    %% 5. Create and Update "Rejected epochs" list
    fid = fopen([path_reports 'rejected_epos.csv'], 'a') ;
    sub_name = strsplit(filename, '_');
    sub_name = [sub_name{1} '_' sub_name{2}];
    epos = strjoin(arrayfun(@(x) num2str(x),disc_epo,'UniformOutput',false),'-');
    c = {sub_name, ...
        sprintf('%i',length(disc_epo)), ...
        sprintf('%s', epos)};
    fprintf(fid, '%s,%s,%s\n',c{1,:}) ;
    fclose(fid);
end


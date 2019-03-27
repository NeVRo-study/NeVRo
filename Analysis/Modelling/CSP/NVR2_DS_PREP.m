%% NVR Downsample and apply PREP
% 2017 by Felix Klotzsche and Alberto Mariola*
% *: main contribution
% inspired by Parra (2004)
%
% References:
% Parra L C, Spence C D, Gerson A D and Sajda P (2005):
% Recipes for the linear analysis of EEG
% NeuroImage 28 326–41
%
% Code from:
% http://www.parralab.org/teaching/eeg/logist.m


%This script loads the raw EEG data to EEGLAB and sets it up for further
%processing

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
% input paths:
path_dataeeg = [path_master 'data/EEG/'];
path_in_eeg = [path_dataeeg 'EEG_raw_cont/SETs/']; % [path_dataeeg 'SBA_PREP/']; %

% output paths:
path_out_eeg = [path_dataeeg 'F_EEG/PREP_continuous/']; % [path_dataeeg 'SBA_PREP_EventsAro/']; % [path_dataeeg 'SBA_raw_EventsAro/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end
path_reports = [path_out_eeg 'reports/'];
if ~exist(path_reports, 'dir'); mkdir(path_reports); end


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
    
    %% 3.Resample to 250 Hz
    % Downsample data.
    NewSamplingRate = 250;
    [EEG, com] = pop_resample(EEG, NewSamplingRate);
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
    
    %% 4. PREP Pipeline
    %EEG = pop_prepPipeline(EEG,struct('ignoreBoundaryEvents', true, 'referenceChannels', [1   2   3   4   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  28  29  30  31  32], 'evaluationChannels', [1   2   3   4   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  28  29  30  31  32], 'rereferencedChannels', [1:32], 'ransacOff', false, 'ransacSampleSize', 50, 'ransacChannelFraction', 0.25, 'ransacCorrelationThreshold', 0.75, 'ransacUnbrokenTime', 0.4, 'ransacWindowSeconds', 5, 'srate', 250, 'robustDeviationThreshold', 5, 'correlationWindowSeconds', 1, 'highFrequencyNoiseThreshold', 5, 'correlationThreshold', 0.4, 'badTimeThreshold', 0.01, 'maxReferenceIterations', 4, 'referenceType', 'Robust', 'reportingLevel', 'Verbose', 'interpolationOrder', 'Post-reference', 'meanEstimateType', 'Median', 'samples', EEG.pnts, 'detrendChannels', [1:32], 'detrendCutoff', 1, 'detrendStepSize', 0.02, 'detrendType', 'High Pass', 'cleanupReference', false, 'keepFiltered', true, 'removeInterpolatedChannels', false, 'lineNoiseChannels', [1:32], 'lineFrequencies', [50  100], 'Fs', 250, 'p', 0.01, 'fScanBandWidth', 2, 'taperBandWidth', 2, 'taperWindowSize', 4, 'pad', 0, 'taperWindowStep', 1, 'fPassBand', [0  125], 'tau', 100, 'maximumIterations', 10, 'reportMode', 'normal', 'publishOn', false, 'consoleFID', 1));
    EEG = pop_prepPipeline(EEG,struct('ignoreBoundaryEvents', true, ...
        'referenceChannels', [1:4, 6:26, 28:32], ...
        'evaluationChannels',[1:4, 6:26, 28:32], ...
        'rereferencedChannels', [1:32], ...
        'ransacOff', false, ...
        'ransacSampleSize', 50, ...
        'ransacChannelFraction', 0.25, ...
        'ransacCorrelationThreshold', 0.75, ...
        'ransacUnbrokenTime', 0.4, ...
        'ransacWindowSeconds', 5, ...
        'srate', 250, ...
        'robustDeviationThreshold', 5, ...
        'correlationWindowSeconds', 1, ...
        'highFrequencyNoiseThreshold', 5, ...
        'correlationThreshold', 0.3, ...
        'badTimeThreshold', 0.01, ...
        'maxReferenceIterations', 4, ...
        'referenceType', 'Robust', ...
        'reportingLevel', 'Verbose', ...
        'interpolationOrder', 'Post-reference', ....
        'meanEstimateType', 'Median', ...
        'samples', EEG.pnts, ....
        'lineNoiseChannels', [1:32], ...
        'lineFrequencies', [50  100], ...
        'Fs', 250, ...
        'p', 0.01, ....
        'fScanBandWidth', 2, ...
        'taperBandWidth', 2, ...
        'taperWindowSize', 4, ...
        'pad', 0, ...
        'taperWindowStep', 1, ...
        'fPassBand', [0  125], ...
        'tau', 100, ...
        'maximumIterations', 10, ...
        'cleanupReference', false, ...
        'keepFiltered', true, ...
        'removeInterpolatedChannels',false, ...
        'reportMode', 'normal', ...
        'publishOn', true, ...
        'sessionFilePath', strcat(path_reports, filename,'_Report.pdf'), ...
        'summaryFilePath', strcat(path_reports, filename,'_Summary.html'), ...
        'consoleFID', 1));
    
    %set(gcf,'Visible','off') ;
    %EEG = pop_saveset( EEG, [filename  '_PREP_Removed.set'] , prep_path);
    s_path = [path_out_eeg 'SBA_PREP/'];
    if ~exist(s_path, 'dir'); mkdir(s_path); end
    EEG = pop_saveset( EEG, [filename  '_PREP.set'] , s_path);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    eeglab redraw;
    
    
    % Find all windows of type figure, which have an empty FileName attribute.
    allPlots = findall(0, 'Type', 'figure', 'FileName', []);
    % Close.
    delete(allPlots);
    
    %% 5. Create and Update "Interpolated Channels" list
    fid = fopen([path_reports 'interpChans_PREP_SBA.csv'], 'a') ;
    c = {filename, ...
        sprintf('%s', num2str(EEG.etc.noiseDetection.interpolatedChannelNumbers)), ...
        sprintf('%i',length(EEG.etc.noiseDetection.interpolatedChannelNumbers))};
    fprintf(fid, '%s,%s,%s,\n',c{1,:}) ;
    fclose(fid);
    
    
end



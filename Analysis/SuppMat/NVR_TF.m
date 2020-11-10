%% NeVRo_TF

% This script computes the power spectrum of both the high and low arousal condition (per single participant). 
% Results (.mat structures) can be fed into NVR_TF_Plot.m for plotting.

%% Clean previous mess

clc
clear all

%% Set paths

% Add general matlab scripts and eeglab plugins
addpath(genpath('/Users/Alberto/MATLAB/'));

% Paths
master_path = '../../../../NeVRo/';
addpath(master_path);

ssd_path = [master_path 'Data/EEG/07_SSD/'];
rat_path = [master_path 'Data/ratings/continuous/not_z_scored/']; 
tf_path = [master_path 'Data/EEG/08.3_TF/']; 

% Folders
cond = {'nomov','mov'};

%%  Loop trough conditions (or selected the preferred one)

for folder = 1:length(cond)
     
rawDataFiles = dir([ssd_path cond{folder} '/SBA/broadband/*.set']);  %we specifcally use SBA data

pwr_str = {};
pwr_str_avg = {};

% Loop trough subjects
for isub = 1:length(rawDataFiles)

loadName = rawDataFiles(isub).name;
fileName = loadName(1:7);

%% Load EEG file

% Open EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Load participant
EEG= pop_loadset([ssd_path cond{folder} '/SBA/broadband/' loadName]);
EEG.setname=fileName;
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%% Let's make sure to only process participants with at least 4 SSD components 

% SSD Selected Components file
ssd_tab = readtable([ssd_path cond{folder} '/SSD_selected_components_' cond{folder} '.csv']); %import the selected components table
%ssd_sel =  str2num(cell2num(ssd_tab{str2num(fileName(6:end)),2})); all together
    
sel_sub = ssd_tab(str2num(fileName(6:end)),2); %select the correspondent participant's row
ssd_sel = str2num(cell2mat(table2array(sel_sub))); %convert selected row to list of numbers (indexes indicating components - Terrible sequence of nested functions but it works)

% Break the loop and start with a new participants if selected ssd
% components are less than <4
if (length(ssd_sel)<4) 
    continue 
end 

%% First remove V/HEOG channels

EEG = pop_select( EEG,'nochannel',{'HEOG' 'VEOG'});
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%% Epoching around markers

EEG = pop_epoch(EEG, {'1','2','3'}, [-0.5 0.5], 'newname', EEG.setname, 'epochinfo', 'yes');

%% Separate dataset(s) based on events

EEG_low = pop_rmdat(ALLEEG(2), {'1'},[-0.5 0.5] ,0);
EEG_high = pop_rmdat(ALLEEG(2), {'3'},[-0.5 0.5] ,0);

%% Compute Power Spectrum on high and low arousal dataset 

[psds_l, freqs_l] = pwelch(EEG_low.data', 250, [], [], EEG.srate);
[psds_h, freqs_h] = pwelch(EEG_high.data', 250, [], [], EEG.srate);

% Compute log power
psds_l_log = log10(psds_l);
psds_h_log = log10(psds_h);

% Compute the mean power spectrum across electrodes
psds_l_mean = mean(psds_l,2);
psds_h_mean = mean(psds_h,2);

psds_l_log_mean = mean(psds_l_log,2);
psds_h_log_mean = mean(psds_h_log,2);

%% Fill the result structure

pwr_str(isub).subj = fileName;
pwr_str(isub).freq = freqs_l;
pwr_str(isub).pwr_high = psds_h;
pwr_str(isub).pwr_low = psds_l;
pwr_str(isub).pwr_high_log = psds_h_log;
pwr_str(isub).pwr_low_log = psds_l_log;
pwr_str(isub).pwr_high_mean = psds_h_mean;
pwr_str(isub).pwr_low_mean = psds_l_mean;
pwr_str(isub).pwr_high_log_mean = psds_h_log_mean;
pwr_str(isub).pwr_low_log_mean = psds_l_log_mean;

%% Plot both spectra (5:48 are related to 4 Hz - 45 Hz)

% Optional: selective simple plot of two participants
% if (folder == 1 && isub == 19) || (folder == 2 && isub == 16)
%     
% figure;
% p1 = plot(freqs_l(1:48),psds_l_log(1:48,:),':','color', 'b');
% hold on
% p2 = plot (psds_l_log_mean(1:48),'color', 'b','LineWidth',2,'DisplayName','Mean (Low)');
% hold on
% p3 = plot(freqs_l(1:48),psds_h_log(1:48,:),':','color', 'r');
% hold on
% p4 = plot(psds_h_log_mean(1:48),'color', 'r','LineWidth',2,'DisplayName','Mean (High)');
% hold off
% legend([p2 p4],{'Mean (Low)','Mean (High)'})
% 
% xlabel('Frequency (Hz)');
% ylabel('Log Power (uV^2)');
% title([fileName]);
% 
% saveas(gcf, [tf_path fileName '_' cond{folder} '_pwr.png']);
% close;
% 
% else
%     ;
% end 

%% Save plot

%saveas(gcf, [tf_path fileName '_' cond{folder} '_pwr.png']);

end 

%% Grand Average and Save structure

% Remove empty rows
pwr_str = pwr_str(all(~cellfun(@isempty,struct2cell(pwr_str))));
str = rmfield(pwr_str,'subj');

% Create Grand Average (bad coding!)
pwr_str_avg.freq = pwr_str(1).freq;
pwr_str_avg.high = mean([str.pwr_high],2);
pwr_str_avg.low = mean([str.pwr_low],2);
pwr_str_avg.log_high = mean([str.pwr_high_log],2);
pwr_str_avg.log_low = mean([str.pwr_low_log],2);
pwr_str_avg.log_high_std = std([str.pwr_high_log_mean],1,2); %
pwr_str_avg.log_low_std = std([str.pwr_low_log_mean],1,2); %

% Save
save([tf_path 'pwr_str_' cond{folder} '.mat'] ,'pwr_str');
save([tf_path 'pwr_str_avg_' cond{folder} '.mat'] ,'pwr_str_avg');

end
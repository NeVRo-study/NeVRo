%% NeVRo_Peak
%This script finds the peak in a selected frequency band for both the
%resting state and SBA task-related recordings.
%Please note: SBA are ICA-cleaned (.set) whereas resting state data are
%raw

%% 0.Clear preceeding mess

clear all
close all
clc

%% 1.Open EEGLAB and set paths

%1.1 Add general matlab scripts and eeglab plugins
addpath(genpath('/Users/Alberto/Documents/MATLAB/'));

%1.2 Open eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

%1.3 Remove PREP pipeline folder (problem with its findpeaks function)
rmpath(genpath('/Users/Alberto/Documents/MATLAB/eeglab14_1_2b/plugins/PrepPipeline0.55.3/'));  %remove PREP pipeline folder (problem with its findpeaks function)

%1.4 Paths
master_path= '/Users/Alberto/Documents/PhD/PhD_Side/NVR';
mov_path= [master_path '/NVR_EEG/rejcomp/mov/SBA/'];
nomov_path= [master_path '/NVR_EEG/rejcomp/nomov/SBA/'];
ssd_path= [master_path '/NVR_EEG/NVR_SSD/'];
rest_path= [master_path '/NVR_EEG/NVR_RS/'];
out_path=[master_path '/NVR_Docs/'];

%% 2.Set parameters (bands, power spectrum etc.)

% 2.1 Power spectrum 
seg_pwr= 10;

% 2.2 Insert band of interest 
low_f=8;
high_f=13;

% 2.3 Folders (tasks) of interests
%Number in files correspond to specific conditons:
%1: resting state
%2: no movement data
%3: movement data

folders= {rest_path, nomov_path, mov_path};  
%% 3.Load data

%Loop trough tasks
for fold = 1:length(folders)

%for fold = 1:1
%3.1 Set names
if   fold==1 
    rawDataFiles = dir([char(folders(fold)) '*_Close.vhdr']);
else fold==2
    rawDataFiles = dir([char(folders(fold)) '*.set']);
end

%3.2 Set empty general peak-matrix
subj_peaks=zeros(length(rawDataFiles),3);

%Loop trough participants
for subi = 1:length(rawDataFiles)

%for subi = 1:1
    
loadName = rawDataFiles(subi).name;
fileName = loadName(1:7);

%2.2 Load (task-related data)

if fold==1 %resting state data are still raw
    [EEG,com] = pop_loadbv([char(folders(fold))],loadName);
    EEG = pop_resample(EEG, 250);
else 
    [EEG,com]= pop_loadset([char(folders(fold)) loadName]);
end 

EEG = eegh(com,EEG);
EEG.setname=fileName;
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

eeglab redraw

%% 3.Find peak in the selected frequency band  - to run this section it's necessary to download the neurospec package (http://www.neurospec.org/) by David M. Halliday.

%3.1 First remove V/HEOG channels
EEG = pop_select( EEG,'nochannel',{'HEOG' 'VEOG'});
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%3.2 Power spectrum 
power_chans=zeros(512,EEG.nbchan+1); %initialize empty matrix (for the ps of each channel)
EEGinv=EEG.data'; %invert data

%Frequencies space
mean_chan=zeros(512,2);

for c= 2:size(EEGinv,2)
    
[f,t,cl]=sp2a2_R2(EEGinv(:,c),EEGinv(:,c),EEG.srate,seg_pwr);

%Frequencies
power_chans(:,1)=f(:,1);
mean_chan(:,1)=f(:,1);

power_chans(:,c)=f(:,2);
plot(power_chans(:,1),power_chans(:,c));
hold on 

mean_chan(:,2)=mean(power_chans(:,2:30),2);
%plot(power_chans(:,1),mean_chan(:,2),'LineWidth',5)
end 

plot(power_chans(:,1),mean_chan(:,2),'LineWidth',5);

% 3.3 Find local peaks in power for the averaged power across channels (NB:
%watch out for conflicting version of findpeaks e.g. the PREP pipeline one)

%band of interest (in mean_chan coord - it depends on the resolution parameter
%.. which is seg_pwr and type of data)
low=low_f*4;
high=high_f*4;

%find peak in the selected band (it should be only one due to the
%MinPeakDistance)
[pks,locs]=findpeaks(mean_chan(low:high,2),mean_chan(low:high,1),'MinPeakDistance',4.5);

%% 4. Insert data in matrix
subj_peaks(subi,1)=single(str2num(fileName(6:end)));
subj_peaks(subi,2)=locs;
subj_peaks(subi,3)=pks;

end

%4.1 Save peaks in both .mat and .csv
save([out_path num2str(fold) '_peaks.mat'],'subj_peaks');
csvwrite([out_path num2str(fold) '_peaks.csv'],'subj_peaks');

end


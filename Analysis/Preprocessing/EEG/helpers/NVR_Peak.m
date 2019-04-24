%% NeVRo_Peak
%This script finds the peak in a selected frequency band for both the
%resting state and SBA task-related recordings.
%Please note: SBA are ICA-cleaned (.set) whereas resting state data are
%raw

%% 1.Open EEGLAB and set paths

% 1.1 Add general matlab scripts and eeglab plugins
addpath(genpath('/Users/Alberto/Documents/MATLAB/'));

% 1.2 Open eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% 1.3 Remove PREP pipeline folder (problem with its findpeaks function) -
%comment out if you don't have it in your path
% e.g:
rmpath(genpath('/Users/Alberto/Documents/MATLAB/eeglab14_1_2b/plugins/PrepPipeline0.55.3/'));  %remove PREP pipeline folder (problem with its findpeaks function)

%Periodogram Parameters - to be put in the function
cropstyle = 'SBA';
seg_pwr = 10;

% 1.4 Paths (always assuming your cd is the one where you opened the script)
path_data = '../../../../Data/';
addpath(path_data);

path_dataeeg  =  [path_data 'EEG/'];
path_eegraw   =  [path_dataeeg '01_raw/rs_raw/'];
path_eegnomov =  [path_dataeeg '06_rejcomp/nomov/' cropstyle '/'];
path_eegmov   =  [path_dataeeg '06_rejcomp/mov/' cropstyle '/'];

% 1.5 Folders (tasks) of interests
folders= {path_eegraw, path_eegnomov, path_eegmov};

for fold = 1:length(folders)
    
    if fold==1
    rawDataFiles_rs = dir([char(folders(fold)) '*_Close.vhdr']);
    elseif fold==2 
    rawDataFiles_nomov = dir([char(folders(fold)) '*.set']);
    else
    rawDataFiles_mov = dir([char(folders(fold)) '*.set']);
    end
end

% 1.5.2 Combine them
rawDataFiles_comb = struct('RS',rawDataFiles_rs, 'NoMov',rawDataFiles_nomov, 'Mov',rawDataFiles_mov);
rawData_cell = struct2cell(rawDataFiles_comb);

% 1.6 Let's look for the max length between the three structures
%[max_l I]=  max(structfun(@(field) length(field),rawDataFiles_comb));
[max_l I]=  max(cellfun(@(field) length(field),rawData_cell));

% 1.7 Set empty general peak-matrix 
subj_peaks=zeros(max_l,length(rawData_cell));

% 1.8 Insert "row header" to always double check the right participant
r_head =  ({rawData_cell{I}.name})'; %row headers 
subj_peaks=[r_head, num2cell(subj_peaks)];  %concatenate them to the sub_peaks matrix

%% 2.Load data

for r = 2:length(rawData_cell)
    
    %Extract current data structure 
    curr_data = rawData_cell(r);
    curr_data = curr_data{1};
    
% Loop trough participants
for isub = 1:length(curr_data)
  
loadName = curr_data(isub).name;
fileName = loadName(1:7);

% 2.2 Load files

if  r==1 %resting state data are still raw
    [EEG,com] = pop_loadbv([(curr_data(r).folder) '/'], loadName);
    EEG = pop_resample(EEG, 250);
else 
    [EEG,com] = pop_loadset([(curr_data(r).folder) '/' loadName]);
end 

EEG = eegh(com,EEG);
EEG.setname=fileName;
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);


%% 3.Find peak in the selected frequency band 

% 3.1 First remove V/HEOG channels
EEG = pop_select( EEG,'nochannel',{'HEOG' 'VEOG'});
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

% 3.2 Power spectrum 
power_chans=zeros(512,EEG.nbchan+1); %initialize empty matrix (for the ps of each channel)
EEGinv=EEG.data'; %invert data

%Frequencies space
mean_chan=zeros(512,2);

for c= 1:size(EEGinv,2)

% NB: sp2a2_R2 from the Neurospec package (http://www.neurospec.org/).
% It computes the periodgram of the signal + a series of measures
% (such as coherence - no need for them here). The function allows for a
% very flexible definition of the power spectrum params. 
[f,t,cl]=sp2a2_R2(EEGinv(:,c),EEGinv(:,c),EEG.srate,seg_pwr);

% Frequencies
power_chans(:,size(power_chans,2))=f(:,1);  %frequency spectrum of the single channel
mean_chan(:,1)=f(:,1);   %put it also in the mean_chan matrix

power_chans(:,c)=f(:,2);   %normalized power at every freq. point
%plot(f(:,1),power_chans(:,c));
%hold on 

mean_chan(:,2)=mean(power_chans(:,1:30),2);
%plot(power_chans(:,1),mean_chan(:,2),'LineWidth',5)
end 

%plot(f(:,1),mean_chan(:,2),'LineWidth',5);

% 3.3 Find local peaks in power for the averaged power across channels 
%(NB: watch out for conflicting version of findpeaks e.g. the PREP pipeline one)

% Band of interest (in mean_chan coord - it depends on the resolution parameter
%.. which is seg_pwr and type of data)
low=low_f*4;
high=high_f*4;

% Find peak in the selected band (it should be only one due to the
% MinPeakDistance)
[pks,locs]=findpeaks(mean_chan(low:high,2),mean_chan(low:high,1),'MinPeakDistance',4.5);

%% 4.Insert data in matrix/cell array

tab_name = char(subj_peaks(isub,1));
i=isub;

%Print 'Nan' if you can't find the specific sub. ID within the complete
%list and if the space is empty
while ((str2num(fileName(6:7)) ~= str2num(tab_name(6:7)))) & (cell2num(subj_peaks(i,r+1)) == 0)
    subj_peaks(i,r+1)={'NaN'}; %print NaN
    i= i+1; %increment isub
    tab_name = char(subj_peaks(i,1)); %update row_header name       
   % end
end
    subj_peaks(i,r+1)=num2cell(locs);
    
end

close all;
end

%% 4.Fix cell array and save

% 4.1 
head = {'participant','rs','nomov','mov'};
output = [head; subj_peaks];

% 4.2 Save peaks in both .mat and .csv
save([out_path num2str(fold) '_peaks.mat'],'output');
csvwrite([out_path num2str(fold) '_peaks.csv'],'output');



%% NeVRo_Peak
% This script finds the peak in a selected frequency band for both the
% resting state and SBA task-related recordings.
% Please note: SBA are ICA-cleaned (.set) whereas resting state data are raw

%% 1.Open EEGLAB and set paths

% 1.1 Add general matlab scripts and eeglab plugins
% NB: Add your general matlab and eeglab paths

% 1.2 Open EEGLAB
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

% Periodogram Parameters - to be put in the function
cropstyle = 'SBA';

% 1.4 Paths (always assuming your cd is the one where you opened the script)
path_data = '../../../../Data/';
addpath(path_data);

path_dataeeg  =  [path_data 'EEG/'];
path_eegraw   =  [path_dataeeg '01_raw/rs_raw/'];
path_eegnomov =  [path_dataeeg '06_rejcomp/nomov/' cropstyle '/'];
path_eegmov   =  [path_dataeeg '06_rejcomp/mov/' cropstyle '/'];
path_eegssd   =  [path_dataeeg '07_SSD/'];

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
subj_peaks=nan(max_l,length(rawData_cell));

% 1.8 Insert "row header" to always double check the right participant
r_head =  ({rawData_cell{I}.name})'; %row headers 
r_head = cellfun(@(x){x(1:7)}, r_head);
subj_peaks=[r_head, num2cell(subj_peaks)];  %concatenate them to the sub_peaks matrix

%% 2.Load data

%Loop trough folders (experimental conditions)
for r = 1:length(rawData_cell)
    
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
    EEG = pop_eegfiltnew(EEG, 1, [], [], 0);%high pass
else 
    [EEG,com] = pop_loadset([(curr_data(r).folder) '/' loadName]);
end 

EEG = eegh(com,EEG);
EEG.setname=fileName;

% 2. Remove V/HEOG channels
EEG = pop_select( EEG,'nochannel',{'HEOG' 'VEOG'});
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

%% 3.Find peak in the selected frequency band 

%Calculate power spectrum of all channels
[psds, freqs] = pwelch(EEG.data', 1250, [], [], EEG.srate);

% Transpose, to make FOOOF inputs row vectors
freqs = freqs';

%Settings of the FOOF function 
settings = struct();
%settings.max_n_peaks = ; 
settings.min_peak_amplitude = 0;
settings.peak_threshold = 2;
settings.peak_width_limits = [1 12];

f_range = [1, 40];

%Run FOOOF across a group of power spectra
fooof_results = fooof_group(freqs, psds, f_range, settings);

%Loop trough channels to extract the max among the single peaks
%(otherwise we will average trough all the small peaks)

peaks = zeros(length(fooof_results),1);

for c=1:length(fooof_results)
    
    %Select it's within a "wider alpha range" (8-13 hz)
    [row, col] = find(fooof_results(c).peak_params(:,1)>=8 & fooof_results(c).peak_params(:,1)<=13);
    
    %Find the maximum between them
    [max_num,max_idx]=max(fooof_results(c).peak_params(row,2));

    % but be sure that it's not absent..
if isempty(row)
    peaks(c) = NaN;
else 
    peaks(c) = fooof_results(c).peak_params(row(max_idx),1); %store the max peak
end
end 

%Average the peaks across channels (excluding NaNs)
peak = mean(peaks,'omitnan');

%METHOD WITHOUT SINGLE CHANNELS CHECK (deprecated for now, but ideal if assuming a
%single peak per channel)
%Extract the peak frequency (mean of all the peaks across channels)
% peak = mean(cat(1, fooof_results(:).peak_params),1);
% peak = peak(1);

%% 4.Insert data in matrix/cell array

tab_name = char(subj_peaks(isub,1));
i=isub;

%Print respective peak if ID within the complete list of participants.
while ((str2num(fileName(6:7)) ~= str2num(tab_name(6:7))))
    i= i+1; %increment isub
    tab_name = char(subj_peaks(i,1)); %update row_header name       
end
    subj_peaks(i,r+1)=num2cell(peak);
    
end

close all;
end

%% 4.Fix cell array and save

% 4.1 Fix col headers
head = {'participant','rs','nomov','mov'};
output = [head; subj_peaks];

% 4.2 Save peaks in both .mat and .csv
%mat
save([path_eegssd 'alphapeaks_FOOOF_fres012_813.mat'],'output');

%csv
T = cell2table(output(2:end,:),'VariableNames',output(1,:));
writetable(T,[path_eegssd 'alphapeaks_FOOOF_fres012_813.csv']);




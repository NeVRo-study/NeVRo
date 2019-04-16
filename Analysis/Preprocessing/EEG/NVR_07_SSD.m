
%% NVR_07_SSD
%
% Run SSD.

% 
% 
% 

% 2019: Alberto Mariola & Felix Klotzsche --- eioe


function NVR_07_SSD(cropstyle, mov_cond, varargin)

% check input:
if nargin > 2
    alphaPeakSource = varargin{1};
else
    alphaPeakSource = mov_cond;
end

%1.1 Set different paths:
path_data = '../../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg '06_rejcomp/' mov_cond '/' cropstyle '/'];
path_in_aPeaks = [path_dataeeg '07_SSD/'];

% output paths:
path_out_eeg = [path_dataeeg '07_SSD/' mov_cond '/' cropstyle '/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

% Get the alpha peaks:

alphaPeaks = load([path_in_aPeaks 'alphaPeaks.mat'], 'alphaPeaks');
alphaPeaks = alphaPeaks.alphaPeaks; % now I would like to throw up.


for isub = (1:length(files_eeg))

    %1.3 Launch EEGLAB:
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = strcat(thissubject, mov_cond, '_PREP_', cropstyle, '_eventsaro_rejcomp');
    filename = char(filename);

    % 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);
    
    % Get the individual alpha peak:
    idx = 
    alphaPeaks(
    
    [ALLEEG EEG] = pop_ssd(ALLEEG,EEG,CURRENTSET,...
                           num2str(locs),...
                           '2',...
                           0,...
                           '15',...
                           1,...
                           0,...
                           0,...
                           '',...
                           []);

end


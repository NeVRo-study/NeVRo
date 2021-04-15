%% NVR_07_1_cutBreak
function NVR_07_1_cutBreak(cropstyle, mov_cond, labelstyle) 
%
% NVR_07_1_cutBreak(cropstyle, mov_cond, labelstyle)
%
%       if cropstyle is set to 'SA' ('Space+Andes') this script
%       cuts out the (30s) EEG data recorded during the break, leaving
%       only data from the two rollercoasters. Importantly, in case
%       categorical/binarized (actually ternarized) rating labels are needed, the have to be 
%       separately loaded to end up with balanced classes again.
%       Requires behavioral arousal rating data in seperate folder 
%       (ratings/...).
%       If cropstyle is 'SBA', this does nothing.
%       Otherwise it saves the processed files in
%       '/Data/EEG/07_SSD/mov_cond/SA/narrowband/'
%   
% arguments:
%       (0) cropstyle: 'SA' or 'SBA'; 'SBA' >>> do nothing
%       (1) mov_cond: either 'mov' or 'nomov'
%       (2) labelstyle (str): continuous or binary (actually ternary) labels
%                             either 'bin(ary)' or 'cont(inuous)';
%                             default: 'bin'

% 2021 by Felix Klotzsche 

%% 1.Set Variables

if strcmp(cropstyle, 'SBA')
    warning('Doing nothing as you chose to keep the break.')
    return
end

%1.1 Set different paths:
path_data = '../../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg '07_SSD/' mov_cond '/SBA/narrowband/' ]; 

% output paths:
path_out_eeg = [path_dataeeg '07_SSD/' mov_cond '/SA/narrowband/' ];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end


%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

for isub = 1:length(files_eeg)
        
    %1.3 Launch EEGLAB:
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};    
    
    %1.5 Set filenames:
    filename = strcat(thissubject, mov_cond, '_PREP_', 'SBA', '_eventsaro_rejcomp_SSD_narrowband'); 
    filename_out = strcat(thissubject, mov_cond, '_PREP_', 'SA', '_eventsaro_rejcomp_SSD_narrowband'); 
    
    filename = char(filename);
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);
    
    %% Cut out break:
    EEG = NVR_S02_cutBreak_postSSD(EEG, thissubject, labelstyle, mov_cond);
    EEG.setname=filename_out;
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset(EEG, [filename_out  '.set'] , path_out_eeg);
    [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
end


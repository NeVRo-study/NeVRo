%% NVR_S02_cutBreak_postSSD


function EEG = NVR_S02_cutBreak_postSSD(EEG, thissubject, labelstyle, mov_cond) 

%
% EEG = NVR_S02_cutBreak_postSSD(EEG, thissubject, labelstyle, mov_cond):
%       cuts out (30s) EEG data recorded during the break, leaving
%       only data from the two rollercoasters. Importantly, in case
%       categorical/binarized (actually ternarized) rating labels are needed, the have to be 
%       separately loaded to end up with balanced classes again.
%       Requires behavioral arousal rating data in seperate folder 
%       (ratings/...)
%   
% arguments: 
%       (1) EEG: NVR EEG data set
%       (2) thissubject (str): subject identifier (e.g., 'NVR_S15') 
%       (3) labelstyle (str): continuous or binary (actually ternary) labels
%                             either 'bin(ary)' or 'cont(inuous)';
%                             default: 'bin'
%       (4) mov_cond: either 'mov' or 'nomov'

% 2021 by Felix Klotzsche 


%% 0. Parse input
if ~contains(EEG.filename, thissubject)
    warning(['Subject name (%s) not in filename (%s).\n' ... 
             'You sure you know what you are doing?'], thissubject, EEG.filename);
end

allowed_labelstyles = {'bin', 'binary', 'cont', 'continuous'};
if ~any(strcmp(allowed_labelstyles, labelstyle))
    error('Label style (%s) not among allowed: %s', labelstyle, strjoin(allowed_labelstyles, ', '));
end

if strcmp(labelstyle, 'binary')
    labelstyle = 'bin';
end
if strcmp(labelstyle, 'continuous')
    labelstyle = 'cont';
end

allowed_movconds = {'mov', 'nomov'};
if ~any(strcmp(allowed_movconds, mov_cond))
    error('Mov cond (%s) not among allowed: %s', mov_cond, strjoin(allowed_movconds, ', '));
end



%% 1.Set Variables

%1.1 Set rating path:
path_data = '../../../Data/';
if strcmp(labelstyle, 'bin')
    label_path = 'class_bins';
    z_scoring = '';
else
    label_path = 'continuous';
    z_scoring = 'z_scored'
end
path_evaro_rat = [path_data 'ratings/' label_path '/' z_scoring '/' mov_cond '/SA/' ];
files = {dir(path_evaro_rat).name};
file_evaro = string(files(contains(files, thissubject)));

% read events:
rats = readmatrix(strcat(path_evaro_rat, file_evaro));
if strcmp(labelstyle, 'cont')
    rats = rats(:,[3,2]);
end

%% 2. Cut out break section:
EEG = pop_select( EEG,'notime', [150 180] );

% overwrite events:
% - overwriting old events ('append', 'no')
EEG = pop_importevent( EEG, ...
    'append','no', ...
    'event',rats, ...
    'fields',{'latency' 'type'}, ...
    'skipline',0, ...
    'timeunit',0.001, ...
    'optimalign','off');

%% NVR_EventsAro
%This script adds the arousal events to the NeVRo EEG data. 

function NVR_06_rejcomp(cropstyle, mov_cond) 

%% 1.Set Variables
%clc
clear EEG

%1.1 Set different paths:
path_data = '../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg 'ICA/' mov_cond '/' cropstyle '/']; 

% output paths:
path_out_eeg = [path_dataeeg 'rejcomp/' mov_cond '/' cropstyle '/'];
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
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filename = strcat(thissubject, mov_cond, '_PREP_', cropstyle, '_eventsaro_ICA'); 
    filename = char(filename);
    
    % check if there is already an outputfile for this participant:
    if exist([path_out_eeg filename '_rejcomp.set'], 'file')
        fprintf('\n\n\n\n')
        resp = [];
        while ~any(strcmp({'y' 'n'},resp)) 
            fprintf(['There is already a file for this subject: \n\n' ...
                     filename '_rejcomp.set \n\n' ...
                     'Do you still want to do this step? \n' ...
                     'Enter "y" for "yes" \n' ...
                     'or "n" in order to skip this subject: \n'])
            resp = input('Confirm with Enter. \n', 's');
        end
        
        if resp == 'n'
            continue
        end
    end
        
                
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);

    %eeg_rank = rank(EEG.data);
    
% eeg_SASICA crashes in loop, so let's do it manually (below)
% defualt parameters (for GUI) can be takenfrom here:
%
%     EEG = eeg_SASICA(EEG, ...
%         'MARA_enable',0, ...
%         'FASTER_enable',0, ...
%         'FASTER_blinkchanname','VEOG', ... 
%         'ADJUST_enable',1, ... 
%         'chancorr_enable',0,...
%         'chancorr_channames','No channel',...
%         'chancorr_corthresh','auto 4',...
%         'EOGcorr_enable',1,...
%         'EOGcorr_Heogchannames','HEOG',...
%         'EOGcorr_corthreshH','auto 4',...
%         'EOGcorr_Veogchannames','VEOG',...
%         'EOGcorr_corthreshV','auto 4',...
%         'resvar_enable',0,...
%         'resvar_thresh',15,...
%         'SNR_enable',0,...
%         'SNR_snrcut',1,...
%         'SNR_snrBL',[-Inf 0] ,...
%         'SNR_snrPOI',[0 Inf] ,...
%         'trialfoc_enable',0,...
%         'trialfoc_focaltrialout','auto',...
%         'focalcomp_enable',1,...
%         'focalcomp_focalICAout','auto',...
%         'autocorr_enable',1,...
%         'autocorr_autocorrint',20,...
%         'autocorr_dropautocorr','auto',...
%         'opts_noplot',0,...
%         'opts_nocompute',0,...
%         'opts_FontSize',14);

    % manual version:
    EEG = SASICA(EEG);
    eeglab redraw;
    % show component activations:
    pop_eegplot( EEG, 0, 1, 1);
    % plot further component properties (mainly to see freqs >50Hz):
    pop_prop(EEG, 0, 1:size(EEG.icaweights, 1));
    
    fprintf('\n\n\n\n\n');
    disp('Use command "dbcont" to continue.') 
    keyboard;
    input('are you done?');
    % SASICA stores the results in base workspace via assignin. 
    % [Info from Niko Busch's pipeline: 
    % https://github.com/nabusch/Elektro-Pipe
    EEG = evalin('base','EEG'); 
    rej_comps = find(EEG.reject.gcompreject);
    
    % remove marked components from data:
    [EEG, com] = pop_subcomp(EEG, rej_comps, 1);
    
    % save which components were removed:
    EEG.etc.rejcomp = rej_comps;
    
    EEG = eegh(com,EEG);
    EEG.setname=filename;
    %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset(EEG, [filename  '_rejcomp.set'] , path_out_eeg);
   % [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        
end
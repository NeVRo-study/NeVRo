%% NVR_06_rejcomp
% 
% Inspirations and code snippets from:
% https://www.aesthetics.mpg.de/fileadmin/user_upload/Services/ProgressFiles/EEGLab_RunICA_PruneData.html
% by R. Muralikrishnan

function NVR_06_rejcomp(cropstyle, mov_cond)

%% 1.Set Variables
%clc
clear EEG

%1.1 Set different paths:
path_data = '../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eegICA = [path_dataeeg 'cleanICA/' mov_cond '/' cropstyle '/'];
path_in_eegFull = [path_dataeeg 'eventsAro/' mov_cond '/' cropstyle '/'];

% output paths:
path_out_eeg_short = [path_dataeeg 'rejcomp/' mov_cond '/' cropstyle '/short/'];
if ~exist(path_out_eeg_short, 'dir'); mkdir(path_out_eeg_short); end
path_out_eeg_full = [path_dataeeg 'rejcomp/' mov_cond '/' cropstyle '/'];
if ~exist(path_out_eeg_full, 'dir'); mkdir(path_out_eeg_full); end

%1.2 Get data files
files_eegICA = dir([path_in_eegICA '*.set']);
files_eegICA = {files_eegICA.name};


for isub = 1:length(files_eegICA)
    
    %1.3 Launch EEGLAB:
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eegICA{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};
    %thissubject = 'NVR_S06';
    
    
    %1.5 Set filename:
    filenameICA = strcat(thissubject, mov_cond, '_PREP_', cropstyle, '_eventsaro_cleanICA');
    filenameICA = char(filenameICA);
    filenameFull = strcat(thissubject, mov_cond, '_PREP_', cropstyle, '_eventsaro');
    filenameFull = char(filenameFull);
    
    % check if there is already an outputfile for this participant:
    if exist([path_out_eeg_full filenameFull '_rejcomp.set'], 'file')
        fprintf('\n\n\n\n')
        resp = [];
        while ~any(strcmp({'y' 'n'},resp))
            fprintf(['There is already a file for this subject: \n\n' ...
                filenameICA '_rejcomp.set \n\n' ...
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
    [EEG, com] = pop_loadset([path_in_eegICA, filenameICA '.set']);
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
    fprintf("This was %s.\n\n", filenameFull);
    input('are you done?');
    % SASICA stores the results in base workspace via assignin.
    % [Info from Niko Busch's pipeline:
    % https://github.com/nabusch/Elektro-Pipe
    EEG = evalin('base','EEG');
    rej_comps = find(EEG.reject.gcompreject);
    SASICAinfo = EEG.reject.SASICA;
    
    % save which components were removed and SASICA info:
    EEG.etc.rejcomp = rej_comps;
    EEG.reject.SASICA = SASICAinfo;
    
    EEG = eegh(com,EEG);
    EEG.setname=filenameICA;
    %[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    EEG = pop_saveset(EEG, [filenameICA  '_rejcomp.set'] , path_out_eeg_short);
    % [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
    
    %% Now add ICA weights and other info to full files:
    
    % load full files:
    [EEGFull, com] = pop_loadset([path_in_eegFull, filenameFull '.set']);
    
    % Add info:
    EEGFull.icawinv = EEG.icawinv;
    EEGFull.icasphere = EEG.icasphere;
    EEGFull.icaweights = EEG.icaweights;
    EEGFull.icachansind = EEG.icachansind;
    EEGFull.icaact = EEGFull.icaweights*EEGFull.icasphere*EEGFull.data;  
    EEGFull.etc.rejcomp = EEG.etc.rejcomp;
    EEGFull.reject.SASICA = EEG.reject.SASICA;
    
    % remove the components also from full data:
    [EEGFull, com] = pop_subcomp(EEGFull, EEGFull.etc.rejcomp, 1);
    
    % Save full file:
    EEGFull = eegh(com,EEGFull);
    EEGFull.setname=filenameFull;
    EEGFull = pop_saveset(EEGFull, [filenameFull  '_rejcomp.set'] , path_out_eeg_full);

    fprintf('\n\n###################################################\n\n');    
    fprintf("We're done with %s.\n\n", filenameFull);
    fprintf('Number of Removed Components ... %d. \n\n', length(rej_comps));
    fprintf('\n\n###################################################\n\n');
    fprintf('Do you want to proceed with another subject?\n');
    input('Confirm with ENTER or terminate with CTRL+C\n');
end

end
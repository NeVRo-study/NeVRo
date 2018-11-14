function NVR_05_1_prep4ICA(cropstyle, mov_cond)

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
path_data = '../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg 'eventsAro/' mov_cond '/' cropstyle '/'];

% output paths:
path_out_eeg = [path_dataeeg 'ICA2/' mov_cond '/' cropstyle '/'];
%if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end
path_reports = [path_out_eeg 'reports/'];
%if ~exist(path_reports, 'dir'); mkdir(path_reports); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};


% Create report file:
%fid = fopen([path_reports 'rejected_epos.csv'], 'a') ;
%fprintf(fid, 'ID,n_epos_rejected,epos_rejected\n') ;
%fclose(fid);


[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;


discarded = {};
discarded_mat = zeros(length(files_eeg),20);
counter = 0;


for isub = 1:length(files_eeg)
    %1.3 Launch EEGLAB:
    
    
    % 1.4 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};
    
    %1.5 Set filename:
    filename = strcat(thissubject, mov_cond, '_PREP_', cropstyle, '_eventsaro');
    filename = char(filename);
    
    %% 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);
    full_eeg_idx = CURRENTSET;
    eeg_rank = rank(EEG.data);
    
    % make copy to clean:
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, CURRENTSET, ...
        'setname', 'too be cleaned');
    tbc_eeg_idx = CURRENTSET;
    
    % prepare some parameters for ICA preparation:
    % Which channels shall be used to determine noisy epochs (which will be
    % discarded for ICA decomp)?
    % We exclude channels that show coactivation with eye-movements as
    % rejection via extreme values is used and eye-movements cause big
    % deflections. However, ICA will do a way better job to discard this noisy,
    % so we actually want to keep epochs with "only" eye-activity. We want to
    % discard channels with strong motor noise or electrode displacement ("huge
    % bursts").
    ICA_prep_ignore_chans = [];
    ICA_prep_ignore = {'HEOG', 'VEOG', 'Fp1', 'Fp2', 'F8', 'F7'};
    for igc = 1:numel(ICA_prep_ignore)
        idx = find(strcmp({EEG.chanlocs.labels}, ICA_prep_ignore{igc}));
        if ~isempty(idx)
            ICA_prep_ignore_chans(end+1) = idx;
        end
    end
    
    ICA_prep_chans = setdiff([1:32], ICA_prep_ignore_chans);
    
    
    % prep for cleaning:
    EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], ...
        'newname', [filename '_epo'], ...
        'epochinfo', 'yes');
    EEG = eegh(com,EEG);
    EEG.setname=[filename '_epo'];
    
    EEG = pop_eegthresh(EEG,1,ICA_prep_chans ,-100,100,-0.5,0.496,0,0);
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);
    
    %example manual winrej matrix
    epoch = find(EEG.reject.rejthresh);
    epochchanind = cell(1,1);
    for run = 1:length(epoch)
        epochchanind{run} = find(EEG.reject.rejthreshE(:,epoch(run)));
    end
    %epochchanind = EEG.reject.rejthreshE;
    
    rejepochcol =  [.3, .95, .95];
    rejepoch = zeros(1,EEG.trials);
    rejepoch(epoch) = ones(1,length(epoch));
    rejepochE = zeros(EEG.nbchan,EEG.trials);
    for i=1:length(find(rejepoch))
        rejepochE(epochchanind{i},epoch(i))=ones(size(epochchanind{i}));
    end
    
    winrej=trial2eegplot(rejepoch,rejepochE,EEG.pnts,rejepochcol);
    
    resp = [];
    while ~any(strcmp({'y' 'n'},resp))
        fprintf(['This is file: \n\n' ...
            filename '.set \n\n' ...
            'Do you want to inspect the data? \n' ...
            'Enter "y" for "yes" \n' ...
            'or "n" in order to go to next subject: \n'])
        resp = input('Confirm with Enter. \n', 's');
    end
    
    if resp == 'y'
        % plot to check result:
        fprintf('Continue with "dbcont" command');
        eegplot(EEG.data, 'winrej',winrej);
        keyboard;
        input('are you done?');
    end
    
    
end





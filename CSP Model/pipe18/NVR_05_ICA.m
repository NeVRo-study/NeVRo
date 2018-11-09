%% NVR_EventsAro
%This script adds the arousal events to the NeVRo EEG data. 

function NVR_05_ICA(cropstyle, mov_cond) 

%% 1.Set Variables
%clc
%clear all

%1.1 Set different paths:
path_data = '../../Data/';
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg 'eventsAro/' mov_cond '/' cropstyle '/']; 

% output paths:
path_out_eeg = [path_dataeeg 'ICA2/' mov_cond '/' cropstyle '/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end
path_reports = [path_out_eeg 'reports/'];
if ~exist(path_reports, 'dir'); mkdir(path_reports); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};


% Create report file:
fid = fopen([path_reports 'rejected_epos.csv'], 'a') ;
fprintf(fid, 'ID,n_epos_rejected,epos_rejected\n') ;
fclose(fid);


discarded = {};
discarded_mat = zeros(length(files_eeg),20);
counter = 0;


for isub = 1:length(files_eeg)
    %1.3 Launch EEGLAB:
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
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
                           
    
    % prep for cleaning:
    EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], ...
        'newname', [filename '_epo'], ...
        'epochinfo', 'yes');
    EEG = eegh(com,EEG);
    EEG.setname=[filename '_epo'];
    
    
    %% Bandpass filter (5-45Hz):
    %EEG = pop_eegfiltnew(EEG, [], 5, 414, true, [], 0);
    EEG = pop_eegfiltnew(EEG, 'locutoff',5);
    %EEG = pop_eegfiltnew(EEG, [], 45, 74, 0, [], 0);
    EEG = pop_eegfiltnew(EEG, 'hicutoff',45);
    
    %% Calculate variance threshhold:
    EEGsize = size(EEG.data);
    n_chan = EEGsize(1);
    n_trials = EEGsize(3);
    tri_var = ones(n_chan,n_trials);
    for tri = 1:n_trials
        for cha = 1:n_chan
            tri_var(cha,tri) = var(EEG.data(cha,:,tri));
        end
    end
    
    avg_per_trial = mean(tri_var,1);
    q5 = prctile(avg_per_trial,5);
    q95 = prctile(avg_per_trial,95);
    %thresh = q95 ;%+ 1 * (q95 - q5);
    %disc_epo = find(avg_per_trial > thresh);
    disc_epo = find(avg_per_trial > 
    disp(filename)
    fprintf('\nTo be discarded:\n');
    disp(disc_epo)
    
    EEG = pop_rejepoch( EEG, disc_epo, 0);
    
    counter = counter+1;
    discarded_mat(counter,1:length(disc_epo)) = disc_epo;
    discarded{counter} = disc_epo;
    
    %% 5. Create and Update "Rejected epochs" list
    fid = fopen([path_reports 'rejected_epos.csv'], 'a') ;
    sub_name = strsplit(filename, '_');
    sub_name = [sub_name{1} '_' sub_name{2}];
    epos = strjoin(arrayfun(@(x) num2str(x),disc_epo,'UniformOutput',false),'-');
    c = {sub_name, ...
        sprintf('%i',length(disc_epo)), ...
        sprintf('%s', epos)};
    fprintf(fid, '%s,%s,%s\n',c{1,:}) ;
    fclose(fid);
    
       

%     % run ICA:
%     EEG = pop_runica(EEG, 'extended',1,'interupt','on','pca',eeg_rank);
%     
%     EEG = eegh(com,EEG);
%     EEG.setname=filename;
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
%     EEG = pop_saveset(EEG, [filename  '_ICA.set'] , path_out_eeg);
%     [ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
        
end
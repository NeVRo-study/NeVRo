function [EEG] = NVR_S01_prep4ICA(EEG,varargin)
%NVR_S01_prep4ICA(): prepares EEG structure for ICA by removing
%particularly noisy "epochs"
%   

% Do you want to manually check:
if nargin>2
    man_check = varargin{2};
    fprintf(['\n\n\n\n' num2str(man_check) '\n\n\n\n']);
else
    man_check = 1;
    fprintf('\n\nnaaa\n\n\n\n');
end


fprintf(['\n\n\n\n' num2str(nargin) '\n\n\n\n'])
% take rejection threshold parameter if given;
if nargin>1
    rejthresh = varargin{1};
    % otherwise set to default (100mV):
else
    rejthresh = 100;
end

% prepare some parameters for ICA preparation:
% Which channels shall be used to determine noisy epochs (which will be
% discarded for ICA decomp)?
% We exclude all channels that show coactivation with eye-movements as
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


% epoch for cleaning:
EEG = pop_epoch( EEG, {'1','2','3'}, [-0.5 0.5], ...
    'epochinfo', 'yes');
% reject epochs with values >threshhold in at least one channel:
EEG = pop_eegthresh(EEG,1,ICA_prep_chans ,-rejthresh,rejthresh,-0.5,0.496,0,0);
%[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG);

% prepare for plotting (thx to eeglablist!)
% This, however, for now is only for checking (!) and not interactive.
% Epochs cannot be added or unmarked manually. This is on purpose in order 
% to increase replicability.
epoch = find(EEG.reject.rejthresh);
epochchanind = cell(1,1);
for run = 1:length(epoch)
    epochchanind{run} = find(EEG.reject.rejthreshE(:,epoch(run)));
end

rejepochcol =  [.3, .95, .95];
rejepoch = zeros(1,EEG.trials);
rejepoch(epoch) = ones(1,length(epoch));
rejepochE = zeros(EEG.nbchan,EEG.trials);
for i=1:length(find(rejepoch))
    rejepochE(epochchanind{i},epoch(i))=ones(size(epochchanind{i}));
end

winrej=trial2eegplot(rejepoch,rejepochE,EEG.pnts,rejepochcol);

if man_check
    resp = [];
else
    resp = 'n';
end

while ~any(strcmp({'y' 'n'},resp))
    fprintf(['Do you want to inspect the data? \n' ...
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

% Do some checks and reject the marked epochs:
EEG = eeg_checkset(EEG);
% Update rejglobal in channel space with rejections from threshold-rej
EEG = eeg_rejsuperpose(EEG, 1, 0, 1, 0, 0, 0, 0, 0);
% Kick out marked epochs: 
EEG = pop_rejepoch(EEG, find(EEG.reject.rejglobal), 0);
EEG = eeg_checkset(EEG);

end


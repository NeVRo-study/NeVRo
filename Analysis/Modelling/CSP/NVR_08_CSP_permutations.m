%% NVR_08_CSP
% 2017/2018/2020 by Felix Klotzsche* and Alberto Mariola
% *: main contribution
%
% This script applies CSP+LDA to classify the (binarized) level of emotional
% arousal from the data and stores the results.
%
%
% Arguments: - cropstyle
%            - mov_cond
%            - source for alpha peak information (defaults to mov_cond)
%            - show plots? (defaults to TRUE)
%
% Requires EEGLAB (e.g., v13.4.4b) and BCILAB (v1.4-devel). 
% Please note that for an exact replication you will need to mildly tweak
% the BCILAB source (please see details in my extensive comment in the code
% below). 
% A numericically exact replication will probably also fail as no fixed
% seed was/is set for the CV split, so you might end up with slightly (!)
% different results. 


function CSP_results = NVR_08_CSP_permutations(cropstyle, mov_cond, varargin)

%% check input:
if nargin > 4 
    if isa(varargin{3}, 'char') || isempty(varargin{3})
        subject_subset = varargin(3);
    end
else
    subject_subset = [];
end

if ((nargin > 3) && (~isempty(varargin{1})) && (logical(varargin{2})))
    plot_results = true;
else
    plot_results = false;
end
if ((nargin > 2) && (~isempty(varargin{1})))
    alphaPeakSource = varargin{1};
else
    alphaPeakSource = mov_cond;
end


%1.1 Set different paths:
% as BCLILAB will change the pwd, I change the relative paths here:
rand_files = dir(); %get files in current dir to get link to folder;
path_orig = rand_files(1).folder;
path_data = [path_orig '/../../../Data/'];
path_dataeeg =  [path_data 'EEG/'];
path_in_eeg = [path_dataeeg '07_SSD/' mov_cond '/' cropstyle '/narrowband/'];
path_in_SSDcomps = [path_dataeeg '07_SSD/' mov_cond '/'];

% output paths:
path_out_eeg = [path_dataeeg '08.7_CSP_3x10f_reg_mcr_smote_0.2cor/' mov_cond '/' cropstyle '/'];
if ~exist(path_out_eeg, 'dir'); mkdir(path_out_eeg); end
path_out_summaries = [path_out_eeg '/summaries/'];
if ~exist(path_out_summaries, 'dir'); mkdir(path_out_summaries); end
path_out_tmp = [path_out_eeg '/tmp/'];
if ~exist(path_out_tmp, 'dir'); mkdir(path_out_tmp); end

%1.2 Get data files
files_eeg = dir([path_in_eeg '*.set']);
files_eeg = {files_eeg.name};

% Get the SSD comp selection table:
SSDcomps_file = dir([path_in_SSDcomps 'SSD_selected_components_*.csv']);
SSDcomps_tab = readtable([path_in_SSDcomps SSDcomps_file.name]);
SSDcomps_tab = table2struct(SSDcomps_tab);
for p = 1:size(SSDcomps_tab,1)
    if p<10
        p_str = ['NVR_S0' num2str(p)];
    else
        p_str = ['NVR_S' num2str(p)];
    end
    SSDcomps_tab(p).participant = p_str;
end

% Launch BCILAB:

bcilab;

% prepare main figures:
h1 = figure('Visible','Off');
h2 = figure('Visible','Off');

nsubs = length(files_eeg);
ncols = 1;
nrows = ceil(nsubs/ncols);

struct_classAccs = [];

isub_rel = 0;
for isub = (1:nsubs)
    
    % 1.3 Get subj name for getting the right event file later:
    thissubject = files_eeg{isub};
    thissubject = strsplit(thissubject, mov_cond);
    thissubject = thissubject{1};
    if (~isempty(subject_subset) && ~ismember(thissubject, subject_subset))
        continue
    end
    
    fprintf(['###########################################\n\n' ...
        'Starting subject: ' thissubject '\n\n' ...
        '###########################################\n\n']);
        
    %1.4 Launch EEGLAB:
    [ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
    
    %1.5 Set filename:
    filename = strcat(thissubject, ...
        mov_cond, ...
        '_PREP_', ...
        cropstyle, ...
        '_eventsaro_rejcomp_SSD_narrowband');
    filename = char(filename);
    
    % 2.Import EEG data
    [EEG, com] = pop_loadset([path_in_eeg, filename '.set']);
    EEG = eegh(com,EEG);
    eeglab redraw;
    
    % Get the individually selected SSD components:
    idx = find(strcmp({SSDcomps_tab.participant}, thissubject));
    SSD_comps =  SSDcomps_tab(idx).selected_comps;
    SSD_comps_arr = str2num(SSD_comps);
    SSD_comps_ch = ['[' SSD_comps ']']; %transform to char
    % Have to save it in base workspace so that bcilab finds it:
    assignin('base','SSD_comps_ch',SSD_comps_ch);
    if (isnan(SSDcomps_tab(idx).n_sel_comps) || ...
            SSDcomps_tab(idx).n_sel_comps < 4)
        warning(['Not enough valid components found for ' thissubject '!']);
        continue
    end
    
    % Get SSD unmixing matrix:
    SSD_W = EEG.etc.SSD.W'; % needs to be transposed for left multiplication
    % Have to save it in base workspace so that bcilab finds it:
    assignin('base','SSD_W',SSD_W);
    
    % Define approach:
    approach_CSP = {'CSP' ...
        'SignalProcessing', { ...
            'Resampling', 'off' ...
            'Projection', { ...
                'ProjectionMatrix', 'SSD_W' ...
                'ComponentSubset', 'SSD_comps_ch'} ...
            'FIRFilter', 'off' ...
            'EpochExtraction', { ...
                'TimeWindow', [-0.5 0.5]}} ...
        'Prediction', { ...
            'FeatureExtraction', { ...
                'ShrinkageCovariance', true, ...
                'PatternPairs', 2} ...
            'MachineLearning', { ...
                'Learner', {'lda', ...
                    'Lambda', 0.2, ... % search([0:0.2:1]), ...
                    'Regularizer', 'auto'}}}};
    
                
    shuffled_events_mat = nvr_shuffle_eventtypes([EEG.event.type], [1,3], 10, 100, 0.4);
    
    for perm=1:size(shuffled_events_mat, 2)
        shuffled_evs = num2cell(shuffled_events_mat(:,perm));
        [EEG.event.type] = shuffled_evs{:};
        pop_saveset(EEG, [filename '_perm_tmp.set'] , path_out_tmp);
        filename_orig = filename;
        filename_tmp = [filename_orig '_perm_tmp'];
        

        % Load/link data SET to BCILAB:
        isub_data = io_loadset([path_out_tmp, filename_tmp '.set']);

        % train the model and extract results of the CV:
        % PLS NOTICE: NOT TO LOOSE EPOCHS AT THE BOUNDARIES OF THE RECORDING,
        % WE CHANGED THE CODE IN BCILAB/bci_train.m TO NOT REQUIRE "SAFETY"
        % MARGINS. THE ACCORDING SETTING CAN BE CHANGED IN LINE 731 OF THE
        % ACCORDING SOURCE (BCILAB/bci_train.m): 
        % IF YOU RUN OUR ANALYSES WITH THE UNMODFIED BCILAB SOURCE, THE RESULTS
        % OF THE CV WILL ONLY VERY SLIGHTLY VARY (mean accuracies changed by 
        % ~0.3% in our tests).
        % BUT YOU WILL END UP WITH SOME FOLDS THAT HAVE <18 SAMPLES IN THE TEST
        % SET. TO EASEN FURTHER STATISTICAL ANALYSES AND THE UNDERSTANDING OF
        % THE READERS, WE DECIDED TO DROP THE MARGINS TO END UP WITH 10x18 TEST
        % SAMPLES. IN OUR CASE DROPPING THE MARGINS SHOULD BE UNPROBLEMATIC AS
        % NO FILTERING IS APPLIED BY BCILAB. 
        [trainloss,model,stats] = bci_train('Data',isub_data, ...
            'Approach',approach_CSP, ...
            'EvaluationScheme', {'subchron', 3, 10}, ... # [10], ... %[3 10]: 3x10-fold CV;  'loo': leave-one-out
            'EvaluationMetric', 'mcr', ...
            'TargetMarkers',{'1','3'});
        %  'OptimizationScheme', {'subchron', 6, 3, 0}, ...
        %  'OptimizationScheme', 3, ...

        % save to .SET files:
%         EEG.etc.CSP.trainloss = trainloss;
%         EEG.etc.CSP.model = model;
%         EEG.etc.CSP.stats = stats;
% 
%         pop_saveset(EEG, [filename '_CSP.set'] , path_out_eeg);

        % Save to global .mat file:

        CSP_A = model.featuremodel.patterns;
        CSP_W = model.featuremodel.filters;
        SSD_A_sel = EEG.etc.SSD.A(:,SSD_comps_arr);
        SSD_W_sel = SSD_W(SSD_comps_arr,:);

        CSP_results.results(isub).participant = thissubject;
        CSP_results.results(isub).trainloss.perm(perm) = trainloss;
        CSP_results.results(isub).model.perm(perm) = model;
        CSP_results.results(isub).stats.perm(perm) = stats;
%         CSP_results.results(isub).weights.CSP_A.perm(perm) = CSP_A;
%         CSP_results.results(isub).weights.CSP_W.perm(perm) = CSP_W;
%         CSP_results.results(isub).weights.SSD_A_sel.perm(perm) = SSD_A_sel;
%         CSP_results.results(isub).weights.SSD_W_sel.perm(perm) = SSD_W_sel;
        CSP_results.chanlocs = EEG.chanlocs;
    end
    save([path_out_summaries 'CSP_results_perm.mat'], 'CSP_results');

%     
%     struct_classAccs(isub).ID = thissubject;
%     struct_classAccs(isub).acc = 1 - trainloss;
 
end
% 
% saveas(h1, [path_out_summaries 'topoplots_comp1.png'], 'png');
% saveas(h2, [path_out_summaries 'topoplots_comp4.png'], 'png');
% 
% table_classAccs = struct2table(struct_classAccs);
% writetable(table_classAccs, [path_out_summaries '\_summary.csv']);

% back to old pwd:
cd(path_orig);
%% NeVRo EEG preprocessing pipeline
%
% Runs the full preprocessing pipeline over all data sets by calling the 
% single helper functions. It does this in seperate iterations for (a) the 
% movement ("mov") vs. the non-movement ("nomov") condition as well as 
% (b) either including ("SBA" := Space Coaster + Break + Andes Coaster) or 
% excluding ("SA") the break.
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 2018 by Felix Klotzsche (contributions by Alberto Mariola)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Add 
NVR_path = genpath('..\..\..\..\NeVRo\Analysis');
addpath(NVR_path);

% Step 00: Divide EEG data SETs into mov and nomov parts:
% [Should not be necessary any more unless modified.]
% NVR_00_cutParts();

m_conds = {'mov' 'nomov'}; %   {'mov'}; % {'nomov'}; %
c_styles = {'SBA', 'SA'}; %{'SBA'}; % {SA}; %

%% Loop over the movement conditions:
for mc=1:numel(m_conds)    
    %% Step 01: Check the data SETs for integrity (all markers present?):
    % [Should not be necessary any more unless modified.]
    %
    % NVR_01_checkSET(m_conds{mc})
    
    %% Step 02: Downsample to 250Hz and run PREP pipeline (Bigdely-Shamlo et
    %          al., 2015) for standardized preprocessing:
    %
    NVR_02_DS_PREP(m_conds{mc})
        
    %% Loop over cropping styles: 
    % (shall break be included [='SBA'] or not [='SA'])
    for cs = 1:numel(c_styles)
        %% Crop relevant parts:
        NVR_03_crop(c_styles{cs},m_conds{mc});
        
        %% Add individual arousal events:
        NVR_04_eventsARO(c_styles{cs},m_conds{mc});%('SBA','mov');
        
        %% Prepare data for ICA and visualize results of rejection:
        % [Not necessary --- only for visualization!]
        % NVR_05_01_prep4ICA(c_styles{cs},m_conds{mc});
        
        %% Run ICA:        
        NVR_05_ICA(c_styles{cs},m_conds{mc}, 'runica');
        
        %% Reject bad ICA components and reproject ICA weights:
        NVR_06_rejcomp(c_styles{cs},m_conds{mc});
    end
end

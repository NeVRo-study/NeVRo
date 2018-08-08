%% NVR Pipeline Felix
% 2017 by Felix Klotzsche




%% 0. Set different paths:
% path to NVR_master:
path_master = 'G:/NEVRO/NVR_master/';
addpath(genpath([path_master 'analysis']))
path_bcilab = 'E:\Felix\BCILAB\BCILAB-devel';
addpath(genpath(path_bcilab)) 

%% 1. Import the continuous data:
%usually not necessary (uncomment if necessary)
%NVR1_import();

%% 2. Downsample to 250Hz and run the PREP pipeline on the data
%usually not necessary and very time consuming (uncomment if necessary)
%NVR2_DS_PREP();

%% 3. Regress out EOG activity:

NVR3_EOGreg();

%%  4. crop to S(B)A;

NVR4_crop_SBA('SBA');

%% 5. Add the target events:

NVR5_eventsARO('SBA');

%% 6. Reject artifacts (automatically):

NVR6_artrej('SBA');

%% 7. Apply SSD:

for filtered = [1 0]
    for cen_freqs = [6 7 8 9 10 11 12 13 14 16 18 20 22]
        
        % params: croppingstyle, central freq for SSD (char), keep filterd (y/n)
        cen_freq = num2str(cen_freqs);
        NVR7_SSD('SBA', cen_freq, filtered);
        
        %% 8. Apply CSP:
        % params: croppingstyle, central freq for SSD (char), kept filterd after SSD (y/n)
        NVR8_CSP('SBA', cen_freq, filtered);
        
    end
end

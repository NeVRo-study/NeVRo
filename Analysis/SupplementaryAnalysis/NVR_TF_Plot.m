%% NeVRo_TF_Plot

% This script reproduce figure S3 of the Supplementary Materials (average power
% spectrum of both mov and nomov condition).

% NB: Some of the aesthetical details/choices were inspired by this blogpost by
% Anne Urai: https://anneurai.net/2016/06/13/prettier-plots-in-matlab/
% The viridis palette is coherent with the group plots that are reported in
% the main text.

%% Set paths
% NB: Add your general matlab and eeglab paths

tf_path = [master_path 'Data/EEG/08.3_TF/']; 

%% Loading power spectrum of both conditionx

nm = load([tf_path 'pwr_str_avg_nomov.mat']);
m  = load([tf_path 'pwr_str_avg_mov.mat']);

%% Plotting

figure;
title(['Average Power Spectrum']);
hold on;

subplot(1,2,1);
hold on;
%colors = cbrewer('qual', 'Set2', 8);
colors = viridis;
colors = colors([150,220],:);

%subplot(4,4,[13 14]);  % plot across two subplots
bl = boundedline(nm.pwr_str_avg.freq(2:48),...
    nm.pwr_str_avg.log_low(2:48), nm.pwr_str_avg.log_low_std(2:48), ...
    nm.pwr_str_avg.freq(2:48), ...
    nm.pwr_str_avg.log_high(2:48), nm.pwr_str_avg.log_low_std(2:48), ...
    'cmap', colors, 'alpha');

xlim([0 max(nm.pwr_str_avg.freq(1:49))]); 
xlabel('Frequency (Hz)'); 
ylabel('Log Power (uV^2/Hz)');
title(['nomov']);

% instead of a legend, show colored text
lh = legend(bl);
legnames = {'Low Emotional Arousal','High Emotional Arousal'};
for i = 1:length(legnames),
    strings{i} = ['\' sprintf('color[rgb]{%f,%f,%f} %s', colors(i, 1), colors(i, 2), colors(i, 3), legnames{i})];
end
lh.String = strings;
lh.Box = 'off';
 
% move a bit closer
lpos = lh.Position;
lpos(1) = lpos(1) - 0.02;
lh.Position = lpos;

subplot(1,2,2);

bl = boundedline(m.pwr_str_avg.freq(2:48),...
    m.pwr_str_avg.log_low(2:48), m.pwr_str_avg.log_low_std(2:48), ...
    m.pwr_str_avg.freq(2:48), ...
    m.pwr_str_avg.log_high(2:48), m.pwr_str_avg.log_low_std(2:48), ...
    'cmap', colors, 'alpha');

xlim([0 max(m.pwr_str_avg.freq(1:49))]); 
xlabel('Frequency (Hz)'); 
ylabel('Log Power (uV^2/Hz)');
title(['mov']);

% Show colored text
lh = legend(bl);
legnames = {'Low Emotional Arousal','High Emotional Arousal'};
for i = 1:length(legnames),
    strings{i} = ['\' sprintf('color[rgb]{%f,%f,%f} %s', colors(i, 1), colors(i, 2), colors(i, 3), legnames{i})];
end
lh.String = strings;
lh.Box = 'off';
 
% move a bit closer
lpos = lh.Position;
lpos(1) = lpos(1) - 0.02;
lh.Position = lpos;

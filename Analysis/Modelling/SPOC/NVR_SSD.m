%% NeVRo_SSD
%This script computes SSD on the ICA cleaned data (see ...).
%SSD is centered on the individualized peak alpha frequency computed for
%each participant (NVR_Peak.m). 
%Users can choose which peak file to use as a reference (eyes-closed
%resting state, no_movement task or movement_task). Eyes-Closed resting
%state (code:1) is suggested. 

%SSD is currently processed manually via examination of the topoplot,power spectrum
%(via MARA) and eigenvalue of each component.
%Finally, the script produces both a .set file of the resulting EEG structure and a .csv of the
%timecourse of each component.

%% 0.Clear preceeding mess

clear all
close all
clc

%% 1.Open EEGLAB and set paths

%1.1 Add general matlab scripts and eeglab plugins
addpath(genpath('/Users/Alberto/Documents/MATLAB/'));

%1.2 Open eeglab
[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;

%1.3 Paths
master_path= '/Users/Alberto/Documents/PhD/PhD_Side/NVR/';
mov_path= [master_path '/NVR_EEG/rejcomp/mov/SBA/'];
nomov_path= [master_path '/NVR_EEG/rejcomp/nomov/SBA/'];
ssd_path= [master_path '/NVR_EEG/NVR_SSD/'];
resting_path= [master_path '/NVR_EEG/NVR_RS/'];
out_path=[master_path '/NVR_Docs/'];

%% 2.Set parameters

% 2.1 Folders (tasks) of interests
%Number in files correspond to specific conditons:
%1: no movement data
%2: movement data

folders= {nomov_path, mov_path};  

% 2.2 Load the chosen peak matrix (resting or no_mov?)
prompt = 'Which peaks do you want to use? (1=Resting State; 2=No_Mov Task)\n';
x = input(prompt,'s')

sel_peaks=load([out_path num2str(x) '_peaks.mat']);
    
%% 3.Load data

%Loop trough tasks
for fold = 2:length(folders)

%3.1 Set names
rawDataFiles = dir([char(folders(fold)) '*.set']);

%3.2 Open the SSD-rejection .csv file  
fid = fopen([out_path num2str(fold) '_Rej_SSD.csv'], 'a') ;
f = {['Subject'],['Rej. Comp. Number'],['Total Rej.'],['Retained Comp. Number'],['Total Retained']};
fprintf(fid,'%s,%s,%s,%s,%s\n',f{1,:});

% 3.3 Create a vector to store all the lambdas/eigenvalues (length of 30 because it's
% highly unlikley that after preprocessing we will have more than 30)
%Lambdas=zeros(length(rawDataFiles),30); 

%3.4 Loop trough participants
for isub = 6:length(rawDataFiles)

loadName = rawDataFiles(isub).name;
fileName = loadName(1:7);

%Insert subject number in the Lambda table
sel_sub=single(str2num(fileName(6:end)));
Lambdas(isub,1)= sel_sub;

%3.5 Load (task-related data)
[EEG,com]= pop_loadset([char(folders(fold)) loadName]);
EEG = eegh(com,EEG);
EEG.setname=fileName;
[ALLEEG,EEG,CURRENTSET] = eeg_store(ALLEEG, EEG);

eeglab redraw

%% 4.Extract individualized alpha peak 

%Check whether the current subject is 
current_isub=isub;
while (sel_sub~=sel_peaks.subj_peaks(current_isub,1))

    current_isub=current_isub+1;
    
end 

locs=sel_peaks.subj_peaks(current_isub,2);  %corresponding peak 

%% 5.Apply SSD for the Alpha band (defined around the peak found above)

%5.1 Save unfiltered EEG data
EEGunfilt=EEG.data;

%5.2 SSD (EEGLAB - Plugin) 
%the parameter '15' doesn't matter because we don't apply any dimensionality reduction before
% '2' is the default filter order
%[ALLEEG EEG] = pop_ssd(ALLEEG,EEG,CURRENTSET,num2str(locs),'2',0,'15',0,0,0,'',[]); %no filtering
[ALLEEG EEG] = pop_ssd(ALLEEG,EEG,CURRENTSET,num2str(locs),'2',0,'15',1,0,0,'',[]); %keep filtered data

EEGfilt=EEG.data; %save the filtered data

EEG.data=EEGunfilt; %we need broadband data to evaluate SSD performance properly
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

%% 6.Examine component via MARA (check power spectrum to detect bumps in the alpha range)
[ALLEEG, EEG, CURRENTSET]= processMARA(ALLEEG,EEG,CURRENTSET, [0,0,1,0,0]) %SSD selection in a semi-automatized fashion


% 6.1 Plotting components' lambda values
%lambda_img=figure();
l=figure();
bar([EEG.dipfit.model.SPoC_lambda]);
title(strcat('Lambda values for subject: ',fileName));
xlabel('Component number');
ylabel('Lambda');
%print(lambda_img,'-djpeg',strcat(eegplotpath, filename, '_lambda.jpeg'));

keyboard; %block for performing manual selection
 
%6.2 Print out components' IDs before removing them
c = {[fileName],[sprintf('%s', num2str(find(EEG.reject.gcompreject)))],...
    [sprintf('%i', length(find(EEG.reject.gcompreject)))],...
    [sprintf('%s', num2str(find(~EEG.reject.gcompreject)))],...
    [sprintf('%i', length(find(~EEG.reject.gcompreject)))]};

fprintf(fid,'%s,%s,%s,%s,%s\n',c{1,:});

%6.2 Remove selected components and store the new EEG
EEG=pop_subcomp(EEG,find(EEG.reject.gcompreject),0);
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);

%6.3 Insert retained lambdas in the correspoding vector
for l=1:length(EEG.dipfit.model)
    
Lambdas(isub,l+1)=EEG.dipfit.model(l).SPoC_lambda(1);

end 
%% 7.Save SSD processed dataset

%7.1 Broad-band (aka non-filtered)
EEG = pop_saveset(EEG, [fileName '_' num2str(fold) '_broad_SSD.set'] , ssd_path);

%7.2 Export the SSD components-timecourses (channel x time / non-filtered)
pop_export(EEG, [ssd_path, fileName,'_SSD_nonfilt_cmp.csv'],'ica','on','transpose','off','time','off','precision',4);

%7.3 Narrow-band (channel x time / filtered)
EEG.data=EEGfilt;
[ALLEEG EEG CURRENTSET] = eeg_store(ALLEEG, EEG);
EEG = pop_saveset(EEG, [fileName '_' num2str(fold)  '_narrow_SSD.set'] , ssd_path);

%7.4 Export the SSD components-timecourses
pop_export(EEG, [ssd_path fileName '_' num2str(fold) '_SSD_filt_cmp.csv'],'ica','on','transpose','off','time','off','precision',4);

close(ancestor(l, 'figure'))

end

%7.5 Save eigenvalues
save([out_path num2str(fold) '_Lambdas.mat'],'Lambdas');
%.csv
csvwrite([out_path num2str(fold) '_Lambdas.csv'],Lambdas);

%close SSD components file
fclose(fid);

end


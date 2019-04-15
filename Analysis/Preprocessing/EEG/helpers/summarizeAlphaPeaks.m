%% Summarize alpha peaks

% assumes the outputs of NVR_Peak to be loaded as "rest", "nomov", "mov"

path_SSD = ['../../../../Data/EEG/07_SSD/'];
allAlpha = [];
for i=1:44
    allAlpha(i,1:2) = rest(i,1:2);
    if (sum(nomov(:,1) == i))
        allAlpha(i,3) = nomov(nomov(:,1) == i,2);
    else
        allAlpha(i,3) = 0;
    end
    
    if (sum(mov(:,1) == i))
        allAlpha(i,4) = mov(mov(:,1) == i,2);
    else
        allAlpha(i,4) = 0;
    end
end



for (i=1:size(allAlpha,1)) 
    
    subNum = allAlpha(i,1);
    if (subNum<10) 
        subNum_str = ['0' num2str(subNum)];
    else
        subNum_str = num2str(subNum);
    end
        
    name = ['NVR_' subNum_str];
    alphaPeaks(i).name = name;
    alphaPeaks(i).restEyesClosed = allAlpha(i,2);
    alphaPeaks(i).nomov = allAlpha(i,3);
    alphaPeaks(i).mov = allAlpha(i,4);
end

% Save to csv:
fID = fopen([path_SSD 'alphaPeaks.csv'], 'w');
fprintf(fID, 'ID,restEyesClosed,nomov,mov\r\n'); 
fprintf(fID,'%i,%f,%f, %f\r\n',allAlpha');
fclose(fID);

% Save as .mat:
save([path_SSD 'alphaPeaks.mat'], 'alphaPeaks');


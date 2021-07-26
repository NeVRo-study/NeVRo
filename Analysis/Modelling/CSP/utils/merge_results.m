% merge data

function CSP_results_perm_merged = merge_results(cropstyle, mov_cond)



    path_data = '../../../Data/EEG/08.8_CSP_3x10f_regauto_auc_smote_1.0cor/';
    path_res = [path_data '/' mov_cond '/' cropstyle '/' 'summaries/'];

    subIDs = dir([path_res '/NVR_*']);


    for isub = 1:length(subIDs)
        subID = subIDs(isub).name;

        perms = dir([subIDs(isub).folder '/' subID '/perm_*']);

        trainloss_perms = zeros(1, length(perms));

        for p = 1:length(perms)
            permNum = strsplit(perms(p).name, '_');
            permNum = str2double(permNum{2});

            results = load([perms(p).folder '/' perms(p).name '/CSP_results.mat']);

            % compensate inconsistent naming:

            if ismember('var', fieldnames(results))
                results.results_perm = results.var;
            end

            % Just to be sure:
            if ~(strcmp(subID, results.results_perm.participant))
                error("You fucked up the subject IDs.")
            end

            if ~(permNum == results.results_perm.perm)
                error("You fucked up perm numbers.")
            end

            trainloss_perms(permNum) = results.results_perm.trainloss;
            CSP_results_perm_merged.results(isub).stats.perm(permNum).targets = results.results_perm.targets;

        end
        CSP_results_perm_merged.results(isub).participant = subID;
        CSP_results_perm_merged.results(isub).trainloss.perm = trainloss_perms;

    end
    
    save([path_res '/CSP_results_perm_merged.mat'], 'CSP_results_perm_merged');
    
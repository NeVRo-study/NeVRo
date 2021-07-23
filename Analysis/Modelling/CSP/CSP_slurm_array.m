function CSP_slurm_array(jobID)



addpath(genpath('/raven/ptmp/fklotzsche/Software/BCILAB/'))
addpath(genpath('/raven/ptmp/fklotzsche/Experiments/Nevro/'))

ids = {};
for i=1:44
    if i <10
        ss = sprintf('NVR_S0%i', i);
    else
        ss = sprintf('NVR_S%i', i);
    end
    ids{end+1} = {ss};
end

id = ids{jobID};
NVR_08_CSP_permutations_batch_array('SA', 'nomov', 'subject_subset', id, 'smote', true)

% parfor i=1:44
%     id = ids{i}
%     NVR_08_CSP_permutations_batch('SA', 'mov', 'subject_subset', id, 'smote', true) 
% end
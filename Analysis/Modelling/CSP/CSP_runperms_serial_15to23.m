addpath(genpath('/raven/ptmp/fklotzsche/Software/BCILAB/'))
addpath(genpath('/raven/ptmp/fklotzsche/Experiments/Nevro/'))

ids = {};
for i=15:23
    if i <10
        ss = sprintf('NVR_S0%i', i);
    else
        ss = sprintf('NVR_S%i', i);
    end
    ids{end+1} = ss;
end

NVR_08_CSP_permutations('SBA', 'mov', 'subject_subset', ids, 'smote', true)

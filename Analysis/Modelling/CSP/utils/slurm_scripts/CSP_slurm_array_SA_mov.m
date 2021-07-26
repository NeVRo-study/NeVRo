function CSP_slurm_array_SA_mov(jobID)

mov_cond = 'mov';

addpath(genpath('/raven/u/fklotzsche/Software/BCILAB/'))
addpath(genpath('/raven/ptmp/fklotzsche/Experiments/Nevro/Analysis/'))


valid_ids_mov = [2,4,6,8,14,18,21,22,25,28,29,31,34,35,36,37,39,42,44];
valid_ids_nomov = [2,3,4,5,6,8,11,13,14,15,17,18,21,22,25,26,28,29,30,34,35,36,37,39,42,44];



if strcmp(mov_cond, 'mov')
    id_nrs = valid_ids_mov;
else
    id_nrs = valid_ids_nomov;
end

if ismember(jobID, id_nrs)
    if jobID <10
        id = sprintf('NVR_S0%i', jobID);
    else
        id = sprintf('NVR_S%i', jobID);
    end
else
   fprintf("Invalid subject number %i for condition %s", jobID, mov_cond)
   error("Invalid subject number")
end

id = {id};


% if strcmp(mov_cond, 'mov')
%    id_nrs = valid_ids_mov;
%else
%    id_nrs = valid_ids_nomov;
% end

% ids = {};
% for i=1:length(id_nrs)
%    id_nr = id_nrs(i);
%    if id_nr <10
%        ss = sprintf('NVR_S0%i', id_nr);
%    else
%        ss = sprintf('NVR_S%i', id_nr);
%    end
%    ids{end+1} = {ss};
% end



% id = ids{jobID};

fprintf("############# Running Job Array of %s\n", id{1});

NVR_08_CSP_permutations_batch('SA', 'mov', 'subject_subset', id, 'smote', true)

% parfor i=1:44
%     id = ids{i}
%     NVR_08_CSP_permutations_batch('SA', 'mov', 'subject_subset', id, 'smote', true) 
% end

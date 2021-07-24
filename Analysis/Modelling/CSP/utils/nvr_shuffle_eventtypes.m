

function ev_types_shuffled = nvr_shuffle_eventtypes(ev_types_seq, ev_types2shuffle, n_blocks, n_perms, max_cor)

evs = ev_types_seq;
y = evs([ismember(evs, ev_types2shuffle)]);
if rem(length(y), n_blocks)
    error('Splitting %i trials into %i equally sized blocks does not work.', length(y), n_blocks);
end
n_max_perms = factorial(n_blocks) - 1;
if n_perms > n_max_perms 
    warning('With %i blocks only %i different permutations are possible (not %i). ', n_blocks, n_max_perms, n_perms)
end
n_perms = min(n_perms, n_max_perms);

bpm = eye(n_blocks);
l_m = {bpm};
tot = zeros(length(y), n_perms);
% set seed for reproducibility:
rand('seed', 1234);
for p=1:n_perms
    fprintf('##########\n');
    searching = true;
    n_max_run = factorial(n_blocks) - 1;
    run_count = 0;
    while searching 
        run_count = run_count + 1;
        if run_count > n_max_run
            error('No solution found. Decrease number of permutations.');
        end
        % rng('shuffle')
        rand_seq = randperm(n_blocks);
        fprintf('Perm#: %i -- Rand seq: %s', p, num2str([rand_seq]));
        fprintf('\n');
        bpm = bpm(rand_seq,:);
        equals = zeros(length(l_m),1);
        for m = 1:length(l_m)
            equals(m) = isequal(l_m{m}, bpm);
        end
        perm_mat = kron(bpm, eye(length(y)/n_blocks));
        if abs(corr(perm_mat * y', y')) > max_cor
            l_m{end+1} = bpm;
            fprintf('corr too high');
            continue;
        end
        if ~any(equals)
            searching = false;
            l_m{end+1} = bpm;
        end

    end

    perm_mat = kron(bpm, eye(length(y)/n_blocks));
    tot(:,p) = perm_mat * y';
end
   
corrs = corr(tot, y');
if ~all(abs(corrs)<max_cor)
    tot = tot(:, abs(corrs)<max_cor);
    warning('Removing %i permutations with correlations to original rating >%.2f.', sum(abs(corrs)>max_cor), max_cor)
end

evs_shuf = evs;
evs_shuf = repmat(evs_shuf', 1, size(tot,2));
evs_shuf([ismember(evs, ev_types2shuffle)], :) = tot;
ev_types_shuffled = evs_shuf;


%% Display autocorr info

vis_acf_info = false;

if vis_acf_info
    acf_lags = 50;
    acs = zeros(acf_lags+1, size(tot,2));
    for j=1:size(tot,2)
        acs(:,j) = autocorr(tot(:,j), acf_lags); 
    end

    mean_acf = mean(acs,2);
    orig_acf = autocorr(evs([evs~=2])', acf_lags);
    figure
    plot([orig_acf, mean_acf])
end

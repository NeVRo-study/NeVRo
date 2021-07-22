function parsave_results(isub, results, path_out_summaries)
    fname = [path_out_summaries 'CSP_results_perm.mat'];  
    if exist(fname, 'file')
       data = load(fname, 'CSP_results');
       CSP_results = data.CSP_results;
    end
    CSP_results.results(isub).participant = results.participant;
    CSP_results.results(isub).trainloss = results.trainloss;
    CSP_results.results(isub).stats = results.stats;
    save([path_out_summaries 'CSP_results_perm.mat'], 'CSP_results');
end
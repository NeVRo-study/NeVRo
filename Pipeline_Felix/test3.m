for i = 1:38
    
    figure(i)
    topoplot(transpose(results.default(i).model.featuremodel.patterns(1,:)), EEG.chanlocs)
end
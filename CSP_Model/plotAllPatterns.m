for (i=1:32)
    
    figure
    topoplot(results.default(i).model.featuremodel.patterns(1,:), results(1).default(1).model(1).featuremodel.chanlocs)
    title(results.default(i).loss);
end
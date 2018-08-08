
spectra = {};
freqs = {};
alpha = {}
for i = 1:28
    [spectra{i},freqs{i}] = spectopo(EEG.data(i,:,:), 0, EEG.srate);
    alphaIdx = find(freqs{i}>8 & freqs{i}<13);
    alpha{i} = mean(10.^(spectra{i}(alphaIdx)/10));
end
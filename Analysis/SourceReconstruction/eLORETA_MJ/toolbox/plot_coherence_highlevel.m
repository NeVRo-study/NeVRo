function plot_coherence_highlevel(sc,data);

if isfield(sc,'plotcoherencepars');
  plot_coherence(data,sc.locs_2D,sc.plotcoherencepars);
else
  plot_coherence(data,sc.locs_2D);
end



return;


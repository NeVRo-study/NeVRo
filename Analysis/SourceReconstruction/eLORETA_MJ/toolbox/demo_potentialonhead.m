addpath ..
load clab_example;
sa=prepare_sourceanalysis(clab_example);


T=prep_v2head(sa.locs_3D,sa.vc{3}.vc);


dips=[0 3 8 1 0 0]; 
v0=forward_general(dips,sa.fp); 
figure;showfield(v0,sa.locs_2D); 
figure;showsurface(sa.vc{3},[],T*v0);


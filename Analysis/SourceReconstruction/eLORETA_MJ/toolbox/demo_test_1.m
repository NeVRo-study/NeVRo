
load clab_example;


%sa=prepare_sourceanalysis(clab_example);

dips=[[3 0 8 0 0 1];[-3 0 8 0 0 1]]; %define two dipoles
v=forward_general(dips,sa.fp); % calculate the fields;
vv=v*[[1 1 ];[1 -1]]; %mix the the fields

 [s,vmax,imax,dip_mom,dip_loc]=SMusic_v1(vv(:,2),sa.V_fine,sa.grid_fine);
 grid_fine_val=[sa.grid_fine,1./(sqrt(1-s(:,1).^2))]; % s(:,1) is the 'distribution of first source
 figure;showmri(sa.mri,[],grid_fine_val);
 grid_fine_val=[sa.grid_fine,1./(sqrt(1-s(:,2).^2))]; % s(:,2) from second source
 figure;showmri(sa.mri,[],grid_fine_val);
 figure;showmri(sa.mri,[],dips); % here we look again at the true solution;





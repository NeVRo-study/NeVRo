
% How the draw potentials on the head 
load clab_example;
sa=prepare_sourceanalysis(clab_example);

load cs_example; power=diag(cs_example);
figure;showfield(power,sa.locs_2D); 

% sa.elec2head is a matrix which maps values 
% on electrodes to values on the head 
% it was calculated (in perpare_souceanalysis by 
%  elec2head=prep_v2head(sa.locs_3D,sa.head.vc,.5));
figure;showsurface(sa.head,[],sa.elec2head*power);


function sa=prepare_sourceanalysys(clab,para);

newlocs=0;

if nargin>1
  if isfield(para,'newlocs')
    newlocs=para.newlocs;
  end
end


[lintrafo,clab_electrodes,chaninds]=clabstandard2clab(clab);
load standard_pars;
fp_eeg_standard.lintrafo=lintrafo;

sa.fp=fp_eeg_standard;

sa.vc=vc_model_standard;
sa.clab_electrodes=clab_electrodes;
load locs_2D_eegstandard;
locs_2D=locs_2d_eegstandard(chaninds,:);
if newlocs==1
  locs_2D=mk_sensors_plane(locs_2D(:,2:3));
end 

sa.locs_2D=locs_2D;

if newlocs==1
  locs_2D=mk_sensors_plane(locs_2D(:,2:3));
end 
   

load vc_cortex_new;
sa.cortex=vc_cortex_new;

load mri
sa.mri=mri;

load grids
sa.grid_coarse=grid_coarse;
sa.grid_medium=grid_medium;
sa.grid_fine=grid_fine;

load Vs 
sa.V_coarse=V_coarse(chaninds,:,:);
sa.V_medium=V_medium(chaninds,:,:);
sa.V_fine=V_fine(chaninds,:,:);



return;
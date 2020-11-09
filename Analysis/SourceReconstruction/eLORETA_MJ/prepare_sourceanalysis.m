function sa=prepare_sourceanalysis(sa,clab,model,para);
% creates a structure needed for source analaysis
% usage: sa=prepare_sourceanalysys(clab,para);
%
% input: 
% sa is the laoded forward model (eg, sa_nyhead).
% clab is a list of N channel names (clab{1} ... clab{N})
%      all channels have to be EEG channels and the names must 
%      be known to this program. (load clab_example for 
%      list of possible names.)
% para is an optional structure where you can set 
%      details. Right now, you can only set para.newlocs=1      
%      which means that the 2D locations are recalculated 
%      for 'head-in-head-plots' (i.e. plots of cross-spectra). 
%      This can be useful if you have fewer channels than 
%      the original 118. 
%
% output:
% sa   complictaed structure which contains everything 
%      you need for source analysis, including a forward model,
%      tesselated cortex and 3 surfaces of the volume 
%      conductor, MRI data, grids for sources within the 
%      brain, precalculated forward solutions for these 
%      grids, and 2D-locations to display  potentials 
%      and, e.g., cross-spectra.         
%


mymodel='curry1';


newlocs=0;

if nargin>1
    mymodel=model;
end
if nargin>3
  if isfield(para,'newlocs')
    newlocs=para.newlocs;
  end
end

% loading the model is slow, so I just do it once.
% mymodelfile=strcat('sa_',mymodel);
% load(mymodelfile);


if length(clab)>0
sa_new=sa;
[lintrafo,clab_electrodes,chaninds,myinds]=clabstandard2clab(clab,sa.clab_electrodes);

nchan=length(chaninds);
locs_3D=zeros(nchan,6);
locs_3D_orig=zeros(nchan,6);

%chaninds=chaninds
if isfield(sa,'fp')
  sa_new.fp=fp_realistic_reduce_channels(sa.fp,chaninds);
  sa_new.fp.lintrafo=eye(length(chaninds)); 
end
  
if isfield(sa,'fp_sphere')
    sa_new.fp_sphere=fp_sphere_reduce_channels(sa.fp_sphere,chaninds);
    sa_new.fp_sphere.lintrafo=eye(length(chaninds)); 

end
    
if isfield(sa,'fp_sphere2')
    sa_new.fp_sphere2=fp_sphere_reduce_channels(sa.fp_sphere2,chaninds);
    sa_new.fp_sphere2.lintrafo=eye(length(chaninds)); 

end

sa_new.locs_3D=sa.locs_3D(chaninds,:);

if isfield(sa, 'locs_3D_orig')
  sa_new.locs_3D_orig=sa.locs_3D_orig(chaninds,:);
end

sa_new.locs_2D=sa.locs_2D(chaninds,:);
% pars.rot=0;  %! 
% sa_new.locs_2D = mk_sensors_plane(sa_new.locs_3D(:,1:3),pars);


for i=1:length(chaninds);
 clab_electrodes{i}=sa.clab_electrodes{chaninds(i)};
end
sa_new.clab_electrodes=clab_electrodes;


if newlocs==1
  sa_new.locs_2D=mk_sensors_plane(sa_new.locs_2D(:,2:3));
end

if isfield(sa, 'head');
  zmin=10;
  sa_new.elec2head=prep_v2head(sa_new.locs_3D(:,1:3),sa_new.head.vc,zmin);
end

if isfield(sa, 'cortex2K');
  if isfield(sa.cortex2K, 'V')
    sa_new.cortex2K.V = sa.cortex2K.V(chaninds, :, :);
  end
end

if isfield(sa, 'cortex5K');
  if isfield(sa.cortex5K, 'V')
    sa_new.cortex5K.V = sa.cortex5K.V(chaninds, :, :);
  end
end

if isfield(sa, 'cortex10K');
  if isfield(sa.cortex10K, 'V')
    sa_new.cortex10K.V = sa.cortex10K.V(chaninds, :, :);
  end
end

if isfield(sa, 'cortex15K');
  if isfield(sa.cortex15K, 'V')
    sa_new.cortex15K.V = sa.cortex15K.V(chaninds, :, :);
  end
end

if isfield(sa, 'cortex75K');
  if isfield(sa.cortex75K, 'V')
    sa_new.cortex75K.V = sa.cortex75K.V(chaninds, :, :);
  end
  if isfield(sa.cortex75K, 'V_fem')
    sa_new.cortex75K.V_fem = sa.cortex75K.V_fem(chaninds, :, :);
  end
  if isfield(sa.cortex75K, 'V_fem_normal')
    sa_new.cortex75K.V_fem_normal = sa.cortex75K.V_fem_normal(chaninds, :);
  end
  if isfield(sa.cortex75K, 'V_bem')
    sa_new.cortex75K.V_bem = sa.cortex75K.V_bem(chaninds, :, :);
  end
  if isfield(sa.cortex75K, 'V_bem_normal')
    sa_new.cortex75K.V_bem_normal = sa.cortex75K.V_bem_normal(chaninds, :);
  end
  if isfield(sa.cortex75K, 'V_she')
    sa_new.cortex75K.V_she = sa.cortex75K.V_she(chaninds, :, :);
  end
  if isfield(sa.cortex75K, 'V_she_normal')
    sa_new.cortex75K.V_she_normal = sa.cortex75K.V_she_normal(chaninds, :);
  end
end

if isfield(sa, 'mesh_gray');
  if isfield(sa.mesh_gray, 'V_fem')
    sa_new.mesh_gray.V_fem = sa.mesh_gray.V_fem(chaninds, :, :);
  end
end

if isfield(sa, 'mesh_brain5K');
  if isfield(sa.mesh_brain5K, 'V_she')
    sa_new.mesh_brain5K.V_she = sa.mesh_brain5K.V_she(chaninds, :, :);
  end
  if isfield(sa.mesh_brain5K, 'V_she_normal')
    sa_new.mesh_brain5K.V_she_normal = sa.mesh_brain5K.V_she_normal(chaninds, :);
  end
end

if isfield(sa, 'mesh_brain10K');
  if isfield(sa.mesh_brain10K, 'V_she')
    sa_new.mesh_brain10K.V_she = sa.mesh_brain10K.V_she(chaninds, :, :);
  end
  if isfield(sa.mesh_brain10K, 'V_se_normal')
    sa_new.mesh_brain10K.V_se_normal = sa.mesh_brain10K.V_se_normal(chaninds, :);
  end
end

if isfield(sa, 'mesh_brain20K');
  if isfield(sa.mesh_brain20K, 'V_she')
    sa_new.mesh_brain20K.V_se = sa.mesh_brain20K.V_she(chaninds, :, :);
  end
  if isfield(sa.mesh_brain20K, 'V_she_normal')
    sa_new.mesh_brain20K.V_se_normal = sa.mesh_brain20K.V_se_normal(chaninds, :);
  end
end

if isfield(sa,'V');
  sa_new.V=sa.V(chaninds,:,:);
end
if isfield(sa,'V_coarse');
  sa_new.V_coarse=sa.V_coarse(chaninds,:,:);
end
if isfield(sa,'V_fine');
  sa_new.V_fine=sa.V_fine(chaninds,:,:);
end
if isfield(sa,'V_cortex');
    sa_new.V_cortex=sa.V_cortex(chaninds,:,:);
end
if isfield(sa,'V_cortex_coarse');
    sa_new.V_cortex_coarse=sa.V_cortex_coarse(chaninds,:,:);
end
if isfield(sa,'V_cortex_perp');
    sa_new.V_cortex_perp=sa.V_cortex_perp(chaninds,:);
end
if isfield(sa,'V_cortex__coarse_perp');
    sa_new.V_cortex_coarse_perp=sa.V_cortex_coarse_perp(chaninds,:);
end
if isfield(sa,'V_cortex10K');
    sa_new.V_cortex10K=sa.V_cortex10K(chaninds,:,:);
end
if isfield(sa,'V_cortex_normals');
    sa_new.V_cortex_normals=sa.V_cortex_normals(chaninds,:);
end
if isfield(sa,'V_cortex10K_normals');
    sa_new.V_cortex10K_normals=sa.V_cortex10K_normals(chaninds,:);
end
if isfield(sa,'V_medium');
    sa_new.V_medium=sa.V_medium(chaninds,:,:);
end
if isfield(sa,'V_xcoarse');
    sa_new.V_xcoarse=sa.V_xcoarse(chaninds,:,:);
end
if isfield(sa,'V_felix');
    sa_new.V_felix=sa.V_felix(chaninds,:,:);
end
if isfield(sa,'V_xcoarse_incortex');
    sa_new.V_xcoarse_incortex=sa.V_xcoarse_incortex(chaninds,:,:);
end
if isfield(sa,'V_coarse_incortex');
    sa_new.V_coarse_incortex=sa.V_coarse_incortex(chaninds,:,:);
end
if isfield(sa,'V_fine_incortex');
    sa_new.V_fine_incortex=sa.V_fine_incortex(chaninds,:,:);
end
if isfield(sa,'V_coarse_sphere2');
    sa_new.V_coarse_sphere2=sa.V_coarse_sphere2(chaninds,:,:);
end
if isfield(sa,'V_coarse_sphere2_inf');
    sa_new.V_coarse_sphere2_inf=sa.V_coarse_sphere2_inf(chaninds,:,:);
end

if nchan<length(clab);
    sa_new.myinds=myinds;
end

sa=sa_new;
end
return;
function sa=prepare_sourceanalysis(clab,model,para);
% creates a structure needed for source analaysis
% usage: sa=prepare_sourceanalysys(clab,para);
%
% input: 
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
if nargin>2
  if isfield(para,'newlocs')
    newlocs=para.newlocs;
  end
end

mymodelfile=strcat('sa_',mymodel);
load(mymodelfile);


if length(clab)>0
sa_new=sa;
[lintrafo,clab_electrodes,chaninds,myinds]=clabstandard2clab(clab,sa.clab_electrodes);

nchan=length(chaninds);
locs_3D=zeros(nchan,6);

%chaninds=chaninds
sa_new.fp=fp_realistic_reduce_channels(sa.fp,chaninds);
sa_new.fp.lintrafo=eye(length(chaninds)); 

if isfield(sa,'fp_sphere')
    sa_new.fp_sphere=fp_sphere_reduce_channels(sa.fp_sphere,chaninds);
    sa_new.fp_sphere.lintrafo=eye(length(chaninds)); 

end
    
if isfield(sa,'fp_sphere2')
    sa_new.fp_sphere2=fp_sphere_reduce_channels(sa.fp_sphere2,chaninds);
    sa_new.fp_sphere2.lintrafo=eye(length(chaninds)); 

end

sa_new.locs_3D=sa.locs_3D(chaninds,:);
sa_new.locs_2D=sa.locs_2D(chaninds,:);
for i=1:length(chaninds);
 clab_electrodes{i}=sa.clab_electrodes{chaninds(i)};
end
sa_new.clab_electrodes=clab_electrodes;


if newlocs==1
  sa_new.locs_2D=mk_sensors_plane(sa_new.locs_2D(:,2:3));
end

zmin=.5;
sa_new.elec2head=prep_v2head(sa_new.locs_3D(:,1:3),sa_new.head.vc,zmin);

sa_new.V_coarse=sa.V_coarse(chaninds,:,:);
sa_new.V_fine=sa.V_fine(chaninds,:,:);
if isfield(sa,'V_medium');
    sa_new.V_medium=sa.V_medium(chaninds,:,:);
end
if isfield(sa,'V_xcoarse');
    sa_new.V_xcoarse=sa.V_xcoarse(chaninds,:,:);
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
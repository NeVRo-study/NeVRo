function sa=prepare_sourceanalysys(clab,para);
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

newlocs=0;

if nargin>1
  if isfield(para,'newlocs')
    newlocs=para.newlocs;
  end
end


[lintrafo,clab_electrodes,chaninds]=clabstandard2clab(clab);

nchan=length(chaninds);
locs_3D=zeros(nchan,6);

load standard_pars;
fp_eeg_standard.lintrafo=lintrafo;
sa.fp=fp_eeg_standard;

for i=1:length(chaninds);
    locs_3D(i,:)=sa.fp.para_sensors_out{chaninds(i)}.senslocs;
end
sa.locs_3D=locs_3D;




sa.vc=vc_model_standard;
for i=1:3
    sa.vc{i}.vc=sa.vc{i}.vc_ori_model;
    sa.vc{i}=rmfield(sa.vc{i},'vc_ori_model');
end

sa.clab_electrodes=clab_electrodes;
load locs_2D_eegstandard;
locs_2D=locs_2D_eegstandard(chaninds,:);
if newlocs==1
  locs_2D=mk_sensors_plane(locs_2D(:,2:3));
end 

sa.locs_2D=locs_2D;


load vc_cortex_new;
sa.cortex=vc_cortex_new;

load mri_standard
sa.mri=mri;

load grids
sa.grid_coarse=grid_coarse;
%sa.grid_medium=grid_medium;
sa.grid_fine=grid_fine;

load Vs 
sa.V_coarse=V_coarse(chaninds,:,:);
%sa.V_medium=V_medium(chaninds,:,:);
sa.V_fine=V_fine(chaninds,:,:);

load head_model1;
sa.head=head;
load npp_model1;
sa.naspalpar=npp;
sa.elec2head=prep_v2head(sa.locs_3D(:,1:3),head.vc,.5);

return;
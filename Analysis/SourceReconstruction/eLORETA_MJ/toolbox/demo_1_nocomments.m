% prepare 
load clab_example;
sa=prepare_sourceanalysis(clab_example);

% load functional data
load cs_example; % this a cross-spectrum at 10 Hz
[u s v]=svd(real(cs_example));field=u(:,2); % construct an example field for demo


%% viewing options
% fields
figure;showfield(field,sa.locs_2D); 

% cross-spectra
figure; showcs(imag(cs_example),sa.locs_2D);

% surfaces
figure;showsurface(sa.cortex);

para.myviewdir=[0 0 1]; % changes the viewpoint 
figure;showsurface(sa.cortex,para);
dips=[[3 -3 9 0 0 1];[3 3 10 0 0 1]]; %difine two dipoles 
figure;showsurface(sa.cortex,[],dips); %plots dipoles with surface plot
figure;showsurface(sa.cortex,[],dips,sa.cortex.vc(:,2)); %coloring the surface

% mri-slices
figure; showmri(sa.mri);

para.orientation='coronal';
figure;showmri(sa.mri,para);

s1=sa.cortex.vc; % all points an a cortex;
s2=randn(1000,4)+repmat([3 3 3 0],1000,1); %hypothetical source with values
s3=randn(1000,4)+repmat([-3 -3 7 0],1000,1);% another hypothetical source with values
s4=sa.locs_3D; %electrode location plus outward normals (treated as dipoles)
figure;showmri(sa.mri,[],s1,s2,s3,s4);


%%  functional calculations  

% forward calculation
dips=[0 3 8 0 0 1]; 
v0=forward_general(dips,sa.fp);
noise=randn(118,1);
v=v0/norm(v0)+noise/norm(noise)/2;
figure;showfield(v,sa.locs_2D); 

% dipole-fit 
ndip=1; 
[dips_fit,res_out,k,field_theo]=dipole_fit_field(v,sa.fp,ndip); 
showmri(sa.mri,[],dips,dips_fit);



%ndip=2; 
%xtype='imag'; 
%[dips_out,c_source,res_out,c_theo]=dipole_fit_cs(v*v',sa.fp,ndip,xtype);


% two dipoles
dips=[[3 -3 9 0 0 1];[3 3 10 0 0 1]];
v=forward_general(dips,sa.fp); % calculate the fields;
noise=randn(118,2);
vv=(v/norm(v,'fro')+noise/norm(noise,'fro')/5); %add some noise

% Music
[s,vmax,imax,dip_mom,dip_loc]=music(vv,sa.V_fine,sa.grid_fine);
grid_fine_val=[sa.grid_fine,1./(1-s(:,1).^2)];
dips_estimate=[dip_loc,dip_mom];
figure;showmri(sa.mri,[],grid_fine_val,dips,dips_estimate);

% Rapmusic:
ns=2; %number of dipoles
[s,vmax,imax,dip_mom,dip_loc]=rapmusic(vv,sa.V_fine,ns,sa.grid_fine);
dips_estimate=[dip_loc,dip_mom];
grid_fine_val=[sa.grid_fine,1./(1-s(:,2).^2)];
figure;showmri(sa.mri,[],grid_fine_val,dips,dips_estimate);

% you can also draw only points above a threshold 
thresh=3;
res=grid_fine_val(grid_fine_val(:,4)>thresh,:);
figure;showmri(sa.mri,[],res);


%  constraining dipoles on the surface 
load Vcortex; 
[s,vmax,imax,dip_mom,dip_loc]=music(vv,Vcortex,sa.cortex.vc); 
figure;showsurface(sa.cortex,[],1./(1-s(:,1).^2),dips,[dip_loc,dip_mom]);



% real data example:
% load pats;va=real(vi(:,10));vb=imag(vi(:,10));vv=[va,vb];

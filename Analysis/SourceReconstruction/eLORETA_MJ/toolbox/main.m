% first we load a list of the names of EEG-channel 
% we had in a given experiment. Your list may ONLY contain 
% EEG channels (not EOG channels or others). Your list 
% must be a subset of the example below.   
load clab_example;

% Preparation of a structre which contains 
% all necessary information apart from functional 
% data. (The latter is the result of your specifc 
% measurement.)  
sa=prepare_sourceanalysis(clab_example);


% plot a potential 
field=randn(118,1); %random potential as an  example 
figure;showfield_general(field,sa.locs_2D); 


% plot cross-spectra,coherence, etc. 
load cs_example;
figure; plot_coherence(real(cs_example),sa.locs_2D);


% plot cortex
figure;show_vc_tri(sa.cortex);


% plot 4 random dipoles plus cortex;
dips=3*randn(4,3)+[zeros(4,2),5*ones(4,1)]; %random locations 
dips=[dips,randn(4,3)]; %locations plus random moments; 
figure;show_vc_tri(sa.cortex,dips); %plot all dipoles with unit moment; 
% here are some modifications:
clear para
para.myviewdir=[0 0 1]; % changes the viewpoint 
para.normalize=0; % without normalization of dipoles 
figure;show_vc_tri(sa.cortex,dips,para);


% make a forward calculation 
dips=[0 3 8 0 0 1]; % defines dipole, location is [0 3 8] in cm
                    % and moment is [0 0 1] nAm. 
v=forward_general(dips,sa.fp); %makes potential v for that dipole 
                               % dips can also be a Mx6 matrix 
                               % then v is a NxM matrix for N channels 
                               % and the ith. column is the  potential
                               % for the i.th dipole, i.e. the i.th 
			       %  row in dips   

%lets look the calculated potential 
figure;showfield_general(v,sa.locs_2D); 

% make a 1-dipole for one the above potential.  
ndip=1; %select number of dipoles 
[a,res_out,k,field_theo]=dipole_fit(v,sa.fp,ndip); %make the fit 
% a is the estimated dipole solution (each row defines one dipole
% as in dips) res_out is the relative error, k is the number of 
% iterations, and field_theo is the potential of the found 
% dipoles. (Of course, you can calculate field_theo also as   
% field_theo=forward_general(a,sa.fp) if a is just one dipole.)

   

% Plot the MRI without sources: 
figure;showmri(sa.mri);

% Here is an example how to change orientation
para.orientation='sagittal';
figure;showmri(sa.mri,para);

% Here is how to show dipoles 
dips=[[0 3 8 0 0 1];[3 3 6 0 1 0]]; 
figure;showmri(sa.mri,para,dips);


% sa.V_fine contains pre-calculated forward 
% solutions on a fine grid with the brain (.5 cm resolution) 
% V_fine is a NxMX3 matrix where V_fine(:,m,i) is 
% the potential of a unit dipole in i.th direction (x,y,z)
% at the m.th point.  
% 
V_fine_ortho=V2Vortho(sa.V_fine); %We here orthonormalize the  
                                  % forward potentials for each 
                                  % grid point. That is a preprocessing
                                  % for an inverse mapping using MUSIV

% next we load an example for functional data. 
% The final output, vv, is an Nx2 matrix consisting 
% of two potential, and the assumption is that 
% the potential of true dipoles is within the 
% span of vv. Here, vv is constrcuted from ISA, but 
% it could also be, e.g., the  first two eigenvectors 
% of a PCA analysis. It can also consist of 
% just one or more than 2 potentials. 
load pats;va=real(vi(:,10));vb=imag(vi(:,10));vv=[va,vb];

% now we calculate the inverse, spacecorr 
% is an indicator of how well a dipole at the 
%  i.th locations fits with the model assumption.
% spacecorr can have more than 1 column.    
spacecorr=Vortho2eigs(vv,V_fine_ortho);

s=spacecorr(:,1);
grid_fine_val=[sa.grid_fine,1./(1-s)];

cpat=sqrt(-1)*(va*vb'-vb*va');
ndip=2;xtype='imag';
[dips,c_source,res_out,c_theo,a,k,astart]=lm_comp_general(cpat,sa.fp,ndip,xtype);
figure;show_vc_tri(sa.cortex,dips);
figure;[hh,hii]=showmri(sa.mri,para,dips);
figure;[hh,hii]=showmri(sa.mri,para,grid_fine_val);

dips=[3 -3 6 0 1 0];v1=forward_general(dips,sa.fp);
dips=[-3 3 7 0 1 0];v2=forward_general(dips,sa.fp);
cov=v1*v1'+v2*v2'+eye(118)*.001;
v=v1+v2;
clear para;para.method='weighted_min_norm';
clear para;para.method='beamformer';para.cov=cov;
Vinv=V2Vinv(sa.V_fine,para);
clear para;para.pinv=1;
clear para;para.pinv=1;[source,source_mag]=min_norm(v,Vinv,para);
grid_fine_val=[sa.grid_fine,source];
para.dipshow='quiver';figure;showmri(sa.mri,para,grid_fine_val);



dips=[0 -5 6 0 1 0];v1=forward_general(dips,sa.fp);
dips=[0 5 6 0 1 0];v2=forward_general(dips,sa.fp);
cov=v1*v1'+v2*v2'+eye(118)*.001;
v=v1+v2;
clear para;para.method='beamformer';para.cov=cov;
Vinv=V2Vinv(sa.V_coarse,para);
clear para;para.pinv=1;[source,source_mag]=min_norm(v,Vinv,para);
grid_coarse_val=[sa.grid_coarse,source_mag];
para.mymarkersize=12;
figure;[hh,hii]=showmri(sa.mri,para,grid_coarse_val); 
 colmax=max(source_mag);colmin=min(source_mag);
tic
for i=1:100;
  v=cos(i*2*pi/20)*v1+sin(i*2*pi/20)*v2+randn(118,1)*.001;
  [source,source_mag]=min_norm(v,Vinv,para);
   update_showmri(hh,source_mag(hii),[colmin colmax],0);drawnow
end
toc




figure;

spacecorr=Vortho2eigs(vv,V_fine_ortho);
s=spacecorr(:,1);
grid_fine_val=[sa.grid_fine,1./(1-s)];
figure;[hh,hii]=showmri(sa.mri,para,grid_fine_val);

  colmax=max(grid_fine_val(:,4));colmin=min(grid_fine_val(:,4));
  mythresh=50;
  update_showmri(hh,grid_fine_val(hii,4),[colmin colmax],mythresh);refresh;

for i=1:100;
spacecorr=Vortho2eigs(vv,V_fine_ortho);
s=spacecorr(:,1);
grid_fine_val=[sa.grid_fine,1./(1-s)];
  colmax=max(grid_fine_val(:,4));colmin=min(grid_fine_val(:,4));
  mythresh=50;
  update_showmri(hh,grid_fine_val(hii,4),[colmin colmax],mythresh);refresh;
end


V_coarse_ortho=V2Vortho(sa.V_coarse); 


spacecorr=Vortho2eigs(vv,V_coarse_ortho);
s=spacecorr(:,1);
grid_coarse_val=[sa.grid_coarse,1./(1-s)];
figure;[hh,hii]=showmri(sa.mri,para,grid_coarse_val);

  colmax=max(grid_coarse_val(:,4));colmin=min(grid_coarse_val(:,4));
  mythresh=50;
  update_showmri(hh,grid_coarse_val(hii,4),[colmin colmax],mythresh);refresh;

for i=1:100;
spacecorr=Vortho2eigs(vv,V_coarse_ortho);
s=spacecorr(:,1);
grid_coarse_val=[sa.grid_coarse,1./(1-s)];
  colmax=max(grid_coarse_val(:,4));colmin=min(grid_coarse_val(:,4));
  mythresh=i;
  update_showmri(hh,grid_coarse_val(hii,4),[colmin colmax],mythresh);drawnow
end

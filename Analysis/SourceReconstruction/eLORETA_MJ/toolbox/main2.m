load clab_example;

 
sa=prepare_sourceanalysis(clab_example);


% plot potential 
field=randn(118,1);
figure;showfield_general(field,sa.locs_2D);


% plot coherence
coh=randn(118,118);
figure; plot_coherence(coh,sa.locs_2D);


% plot cortex
figure;show_vc_tri(sa.cortex);
figure;show_vc_tri(sa.vc{3});

% plot 4 random dipoles in cortex;
dips=3*randn(4,3)+[zeros(4,2),5*ones(4,1)];
dips=[dips,randn(4,3)];
figure;show_vc_tri(sa.cortex,dips);


% make a forward calculation 
dips=[0 3 8 0 0 1];
v=forward_general(dips,sa.fp);
figure;showfield_general(v,sa.locs_2D);

% 
ndip=1;
[a,res_out,k,field_theo]=dipole_fit(v,sa.fp,ndip);
figure;show_vc_tri(sa.cortex,dips);


para.orientation='sagittal';
figure;showmri(sa.mri,para);
para.orientation='axial';
showmri(sa.mri,para);

V_fine_ortho=V2Vortho(sa.V_fine);
load pats;
va=real(vi(:,10));
vb=imag(vi(:,10));
vv=[va,vb];

cpat=sqrt(-1)*(va*vb'-vb*va');
ndip=2;xtype='imag';
[dips,c_source,res_out,c_theo,a,k,astart]=lm_comp_general(cpat,sa.fp,ndip,xtype);
figure;show_vc_tri(sa.cortex,dips);
figure;[hh,hii]=showmri(sa.mri,para,dips);


spacecorr=Vortho2eigs(vv,V_fine_ortho);
s=spacecorr(:,1);
grid_fine_val=[sa.grid_fine,1./(1-s)];
figure;[hh,hii]=showmri(sa.mri,para,grid_fine_val);

  colmax=max(grid_fine_val(:,4));colmin=min(grid_fine_val(:,4));
  mythresh=50;
  update_showmri(hh,grid_fine_val(hii,4),[colmin colmax],mythresh);refresh;

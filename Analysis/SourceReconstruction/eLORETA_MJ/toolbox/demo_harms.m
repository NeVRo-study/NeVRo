addpath ..
load clab_example;
load ../../surfaces/skin
sa=prepare_sourceanalysis(clab_example);

vcx=skin.vc;

trix=sa.vc{1}.tri;
vcx=sa.vc{1}.vc(:,1:3);
[np ndum]=size(vcx);
vcx=vcx-repmat(mean(vcx),np,1);
rads=sqrt(sum(vcx.^2,2));
vcx=vcx./repmat(rads,1,3);

sphere.vc=vcx;
sphere.tri=trix;
figure;showsurface(sphere);

y=legs_ori(sphere.vc,8);
figure;showsurface(sphere,[],y(:,2));


sph=sphere;sph.vc=sph.vc.*repmat(y(:,5),1,3);figure;showsurface(sphere);


addpath c:\nolte\matlab\eeg\progs


figure;showsurface(skin_b);
locs_3D_onhead=mk_vcharm(sa.locs_3D(:,1:3),center,coeffs);

T=prep_v2head(sa.locs_3D(:,1:3),skin_b.vc,1);

dips=[0 3 8 1 0 0]; 
v0=forward_general(dips,sa.fp); 
%figure;showfield(v0,sa.locs_2D); 
figure;showsurface(skin_b,[],T*v0);


shift=[0 0 -.7];
skin_c=skin_b;[np ndum]=size(skin_c.vc);
skin_c.vc=skin_c.vc+repmat( shift,np,1);
figure;
subplot(2,2,1);
para.myviewdir=[1 0 0];
showsurface(skin_c,para,sa.locs_3D(:,1:3),sa.naspalpar(1:2,1:3));
subplot(2,2,2);
para.myviewdir=[0 1 0];
showsurface(skin_c,para,sa.locs_3D(:,1:3),sa.naspalpar(:,1:3));
subplot(2,2,3);
para.myviewdir=[0 0 1];
showsurface(skin_c,para,sa.locs_3D(:,1:3),sa.naspalpar(:,1:3));







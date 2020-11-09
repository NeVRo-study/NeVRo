addpath c:\nolte\matlab\toolbox
addpath c:\nolte\matlab\toolbox_stefan

clab=importdata('names.txt');
clab=clab';
pat1=importdata('pat1.txt');
pat2=importdata('pat2.txt');
sa=prepare_sourceanalysis(clab,'montreal');
V=reshape(detrend(reshape(sa.V_fine,19,[]),'constant'),19,17177,3);

%figure;
pat=pat1;
[nchan npat]=size(pat);
for i=1:3;
    i=i
    %subplot(4,4,i)
    %showfield(pat2(:,i),sa.locs_2D);colorbar off;



[s,vmax,imax,dip_mom,dip_loc]=music(pat(:,i),V,sa.grid_fine);




[fitg,vmodel]=fitguete(pat(:,i),vmax)

  
figure;
subplot(2,2,1);
  showfield(pat(:,i),sa.locs_2D);colorbar off;
subplot(2,2,2);
  showfield(vmodel,sa.locs_2D);colorbar off;
  text(-.7,.6,num2str(fitg));

  %print -dpsc -append ff1b.ps  
  %close
  st=1./(1-s);
  st4=[sa.grid_fine,st];
  mst=max(st);
  st5=st4(st>.8*mst,:);
  clear para;
  para.orientation='axial';
  figure;showmri(sa.mri,para,st5);
  para.orientation='all';
  para.mricenter=dip_loc;
  figure;showmri(sa.mri,para,st5);
  %print -dpsc -append ff1b.ps  
  %close  
end



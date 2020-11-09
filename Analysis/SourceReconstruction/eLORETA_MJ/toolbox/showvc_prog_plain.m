function showvc_prog_plain(loc,tri,view_dir,para);

cmap=1;
voxelkont=0;
mycolormap='jet';
colorbars=1;
if nargin>3
  if isfield(para,'cmap')
     cmap=para.cmap;
  end
  if isfield(para,'voxelfield')
     voxelfield=para.voxelfield;
     voxelkont=1;
  end
 if isfield(para,'mycolormap')
     mycolormap=para.mycolormap;
 end
 if isfield(para,'colorbars')
     colorbars=para.colorbars;
 end

end




[ntri,ndum]=size(tri);


%colormap(newmap(30:150,:));
brighten(-.6);


locm=mean(loc);
[nloc,ndum]=size(loc);
relloc=loc-repmat(locm,nloc,1);
dis=(sqrt(sum((relloc.^2)')))';thresh=3;dis(dis<thresh)=thresh;
%map=colormap('jet');newmap=colormap_interpol(map,3);size(newmap);colormap(
%newmap);
cortexcolor=[255 213 119]/255;
cortexcolor=[.6 .6 .6];
% cortexcolor=[234 183 123]/255;
h=patch('vertices',loc,'faces',tri);
set(h,'FaceColor',cortexcolor);
view(view_dir);
set(h,'edgecolor','none');
set(h,'facelighting','phong');
%set(h,'specularexponent',50);
set(h,'specularstrength',1);
set(h,'ambientstrength',.6);
set(h,'diffusestrength',.8)
dis0=0*dis+0;
set(h,'facevertexalphadata',dis0)
set(h,'alphadatamapping','direct')
%set(h,'facealpha',.1)
camlight('headlight','infinite');
axis equal 

if voxelkont>0
   vmax=max(voxelfield);vmin=min(voxelfield);
     if vmax <=0 
         vv=vmin;
     else
         vv=vmax;
     end
   tri_new=[];
   nt=length(tri);
   vfx=voxelfield;
     for i=1:nt;
       %xx=0;for j=1:3; xx=xx+norm(voxelfield(tri(i,j)));end;
       if length(voxelfield) == size(loc, 1)
           xx1=min(abs(voxelfield(tri(i,1:3))));
           xx2=mean(abs(voxelfield(tri(i,1:3))));
       else
           xx1=min(abs(voxelfield(i)));
           xx2=xx1;       
       end
       if xx1>0 
         tri_new=[tri_new;tri(i,:)];
       end
       if xx1 ==0 
         vfx(i)=NaN;
       end 
     end
%      vf=voxelfield(abs(voxelfield)>=0);
%      map=colormap(mycolormap);
%      newmap=colormap_interpol(map,3);size(newmap);colormap(newmap);
%      [nc,nx]=size(newmap);
%      vfint=ceil((nc-1)*((voxelfield-min(vf))/(max(vf)-min(vf))));
%      vfint(voxelfield==0)=1;
%      vftruecolor=newmap(vfint);
   %h=patch('vertices',loc,'faces',tri_new,'FaceVertexCData',voxelfield,...
   % 'facecolor','interp','edgecolor','none','facelighting','phong');
%     h=patch('vertices',loc,'faces',tri_new,'FaceVertexCData',vftruecolor,...
%     'facecolor','interp','edgecolor','none','facelighting','phong');
% keyboard
    h=patch('vertices',loc,'faces',tri_new,'FaceVertexCData',vfx,...
    'facecolor','interp','edgecolor','none','facelighting','phong');
   map=colormap(mycolormap);
   newmap=colormap_interpol(map,3);size(newmap);colormap(newmap);
   
   if colorbars==1  
       colorbar;
   end
       
end

set(h,'specularexponent',50000);

return;
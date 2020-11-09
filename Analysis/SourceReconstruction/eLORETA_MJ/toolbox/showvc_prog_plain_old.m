function showvc_prog_plain(loc,tri,view_dir,para);

cmap=1;
voxelkont=0;
if nargin>3
  if isfield(para,'cmap')
     cmap=para.cmap;
  end
  if isfield(para,'voxelfield')
     voxelfield=para.voxelfield;
     voxelkont=1;
  end

end


colp=loc(:,3);
[np,ndum]=size(colp);
colp=loc*view_dir';
[cs,is]=sort(colp);
d=max(colp)-min(colp);
colp(is(1))=colp(is(1))-d/4;
colp(is(2))=colp(is(np))+d/4;

[np,ndum]=size(loc);
loc_b=loc-repmat(mean(loc),np,1);
cov=inv(loc_b'*loc_b);
dis=zeros(np,1);
for i=1:np
    dis(i)=(sqrt(loc_b(i,:)*cov*loc_b(i,:)'));
end
%dis=sqrt(loc_b(:,1).^2+loc_b(:,2).^2+loc_b(:,3).^2);
colp=dis;


%colp(1)=zmin-dz/2;colp(2)=zmax+dz/2;
%cmap='jet';
%map=colormap('gray');
map=colormap('bone');

  colp=colp*cmap;


newmap=colormap_interpol(map,3);size(newmap);colormap(newmap);
%colormap(newmap(30:150,:));
brighten(-.6);

h=patch('vertices',loc,'faces',tri,'FaceVertexCData',colp,...
    'facecolor','interp','edgecolor','none','facelighting','phong');

set(h,'SpecularStrength',0.0,'DiffuseStrength',0);
set(h,'AmbientStrength',1.0);
h=camlight('headlight','infinite');
view(3)
axis equal 
view(view_dir')
%view(3)
colormap(newmap);

if voxelkont>0
   tri_new=[];
   nt=length(tri);
     for i=1:nt;
       xx=0;for j=1:3; xx=xx+norm(voxelfield(tri(i,j)));end;
       if xx>0 
         tri_new=[tri_new;tri(i,:)];
       end 
     end
   h=patch('vertices',loc,'faces',tri_new,'FaceVertexCData',voxelfield,...
    'facecolor','interp','edgecolor','none','facelighting','phong');
   map=colormap('jet');
   newmap=colormap_interpol(map,3);size(newmap);colormap(newmap);
 
end

return;
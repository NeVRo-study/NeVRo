
load clab_example;sc=prepare_sourcereconstruction_highlevel(clab_example);


loc=sc.cortex.vc;
tri=sc.cortex.tri;

locm=mean(loc);
[nloc,ndum]=size(loc);
relloc=loc-repmat(locm,nloc,1);
dis=(sqrt(sum((relloc.^2)')))';thresh=3;dis(dis<thresh)=thresh;
%map=colormap('jet');newmap=colormap_interpol(map,3);size(newmap);colormap(newmap);
cortexcolor=[255 213 119]/255;
cortexcolor=[255 213 119]/255;
cortexcolor=[.6 .6 .6];
view_dir=[-1 0 0];
figure;
%h=patch('vertices',loc,'faces',tri,'facevertexcdata',dis);set(h,'FaceColor','interp');
h=patch('vertices',loc,'faces',tri);set(h,'FaceColor',cortexcolor);
view(view_dir);
set(h,'edgecolor','none');
set(h,'facelighting','phong');
set(h,'specularexponent',50);
set(h,'specularstrength',1);
set(h,'ambientstrength',.6);
set(h,'diffusestrength',.8)
set(h,'facevertexalphadata',dis)
camlight('headlight','infinite');
%vn=get(h,'vertexnormals');
axis equal 
%map=colormap('jet');newmap=colormap_interpol(map,3);size(newmap);colormap(newmap);
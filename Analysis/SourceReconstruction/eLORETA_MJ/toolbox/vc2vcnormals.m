function vc_out=vc2vcnormals(vc);

loc=vc.vc;
tri=vc.tri;

[nloc,ndum]=size(loc);
[ntri,ndum]=size(tri);

locm=mean(loc);
faceori=zeros(ntri,3);
vertexori=zeros(nloc,3);
for i=1:ntri;
  v1=loc(tri(i,1),:)'; v2=loc(tri(i,2),:)'; v3=loc(tri(i,3),:)';
  vm=(v1+v2+v3)/3-locm';
  ori=cross((v1-v2),(v1-v3));
  if ori'*vm<0;ori=-ori;end;
  faceori(i,:)=ori'/norm(ori); 
  for j=1:3;
    vertexori(tri(i,j),:)=vertexori(tri(i,j))+faceori(i,:);
   end  
end

for i=1:nloc;
  vertexori(i,:)=vertexori(i,:)/norm(vertexori(i,:));
end


vc_out=vc;
vc_out.faceori=faceori;
vc_out.vertexori=vertexori;

return;




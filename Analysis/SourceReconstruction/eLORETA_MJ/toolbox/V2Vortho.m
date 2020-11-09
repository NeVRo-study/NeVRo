function Vortho=V2Vorth(V);
% 

[ns,ng,ndum]=size(V);

Vortho=V;

for i=1:ng;
 Vloc=squeeze(V(:,i,:));
 Vortho(:,i,:)=orth(Vloc);
end

return;


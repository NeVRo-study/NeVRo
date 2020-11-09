function spacecorr=Vortho2eigs(patt,Vortho);

[nchan,npat]=size(patt);
patt_ortho=orth(patt);



[nchan,ng,ndum]=size(Vortho);
npatt_effective=min([npat ndum]);
spacecorr=zeros(ng,npatt_effective);
for i=1:ng;
  Vortholoc=squeeze(Vortho(:,i,:));
  A=patt_ortho'*Vortholoc;
  [u,s,v]=svd(A);
  sd=diag(s);
  spacecorr(i,:)=sd(1:npatt_effective)';
end

return;


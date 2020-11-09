function [spacecorr,evsource,evcomp]=V2eigs(patt,V);

[nchan,npat]=size(patt);
[nchan,ng,ndum]=size(V);
np=min([ndum npat]);

[u_p,s_p,v_p]=svd(patt);

 o_patt=v_p*inv(s_p(1:npat,:));


spacecorr=zeros(ng,np);
evsource=zeros(ng,ndum,np);
evcomp=zeros(ng,npat,np);
for i=1:ng;
  Vloc=squeeze(V(:,i,:));
  [u_V,s_V,v_V]=svd(Vloc); 
  o_V=v_V*inv(s_V(1:ndum,:));
  A=u_p(:,1:npat)'*u_V(:,1:ndum);
  [u,s,v]=svd(A); 

  sd=diag(s);
  spacecorr(i,:)=sd(1:np)';
  evsub=o_V*v(:,1:np);
  for j=1:np;
      evsub(:,j)=evsub(:,j)/norm(evsub(:,j));
  end;
  evsource(i,:,:)=evsub;
    evsub=o_patt*u(:,1:np);
  for j=1:np;
      evsub(:,j)=evsub(:,j)/norm(evsub(:,j));
  end;
  evcomp(i,:,:)=evsub;
end

return;


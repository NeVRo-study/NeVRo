function M = beamformer_forooz(L,C,alpha);

if nargin<3
    alpha=0;
end
[nchan ns ndum]=size(L);

c=C+alpha*eye(nchan);
cinv=inv(c);
cinv2=cinv*cinv;
for i=1:ns;
    Lnew=reshape(L(:,i,:),nchan,3);
    K1=transpose(Lnew)*cinv*Lnew;
    K2=transpose(Lnew)*cinv2*Lnew;
    A=inv(K2)*K1;
    M(i)=max(eig(A));
end

    
return;

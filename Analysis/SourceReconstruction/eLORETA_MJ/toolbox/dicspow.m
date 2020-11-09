function po=dicspow(L,C,alpha);

if nargin<3
    alpha=0;
end
[nchan nchan]=size(C);
[nchan ns ndum]=size(L);
Cr=C+alpha*eye(nchan);

Crinv=inv(Cr);

A=zeros(nchan,ns,3);
for i=1:ns
    Lloc=squeeze(L(:,i,:));
    A(:,i,:)=reshape((inv((Lloc'*Crinv*Lloc))*Lloc'*Crinv)',nchan,3);
end


po=zeros(ns,1);
for i=1:ns
    Aloc=transpose(squeeze(A(:,i,:)));
    Ploc=Aloc*C*Aloc';
    [u s v]=svd(Ploc);
    po(i)=s(1,1);
end


    
return;

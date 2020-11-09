function coh=dicscoh(L,C,refind,alpha);

if nargin<4
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

Aref=transpose(squeeze(A(:,refind,:)));
Pref=Aref*C*Aref';
[u s v]=svd(Pref);
Pref=s(1,1);

coh=zeros(ns,1);
for i=1:ns
    Aloc=transpose(squeeze(A(:,i,:)));
    Ploc=Aloc*C*Aloc';
    [u s v]=svd(Ploc);
    Ploc=s(1,1);
    Csloc=Aref*C*Aloc';
    [u s v]=svd(Csloc);    
    Csloc=s(1,1); 
    coh(i)=Csloc/sqrt(Ploc*Pref); 
end


    
return;

function A=mkfilt_eloreta(L, gamma);

% Code by Guido Nolte with minor modifications by Stefan Haufe

[nchan ng ndum]=size(L);
LL=zeros(nchan,ndum,ng);
for i=1:ndum;
    LL(:,i,:)=L(:,:,i);
end
LL=reshape(LL,nchan,ndum*ng);

u0=eye(nchan);
W=reshape(repmat(eye(ndum),1,ng),ndum,ndum,ng);
Winv=zeros(ndum,ndum,ng);
winvkt=zeros(ng*ndum,nchan);
kont=0;
kk=0;
while kont==0;
    kk=kk+1;
    for i=1:ng;
        Winv(:,:,i)=(inv(W(:,:,i)));
        %if i==ng;disp(W(:,:,i));end
    end
    for i=1:ng;
        %winvkt(i,:,:)=Winv(:,:,i)*(squeeze(LL(:,:,i)))';
        %winvkt(i,:,:)=(squeeze(LL(:,:,i)))';
        winvkt(ndum*(i-1)+1:ndum*i,:)=Winv(:,:,i)*LL(:,ndum*(i-1)+1:ndum*i)';
    end
    kwinvkt=LL*winvkt;
    %kwinvkt(1:4,1:4)
%         alpha=.001*trace(kwinvkt)/nchan;
        alpha=gamma*trace(kwinvkt)/nchan;
        M=inv(kwinvkt+alpha*u0);
        
        for i=1:ng;
        Lloc=squeeze(L(:,i,:));
        Wold=W;
        W(:,:,i)=sqrtm(Lloc'*M*Lloc);
        end
    reldef=(norm(reshape(W,[],1)-reshape(Wold,[],1))/norm(reshape(Wold,[],1)));
%     disp(reldef)
    if kk>20 | reldef< .000001 ; kont=1;end;
end
%disp(kk)

ktm=LL'*M;
%ktm=reshape(ktm,ng,ndum,nchan);
 A=zeros(nchan,ng,ndum);

 for i=1:ng;
     A(:,i,:)=(Winv(:,:,i)*ktm(ndum*(i-1)+1:ndum*i,:))';
 end
return
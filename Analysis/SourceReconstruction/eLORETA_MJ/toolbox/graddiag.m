function   [v,diags,parout]=graddiag(M,para);
% approximately diagonalizes arbitrary complex matrices stored in M(:,:,f) for all f
% simplest version:
% Input: M NxNxK tensor with N channels and K matrices to be 
%          diagonalized
%        para is an optional structure. 
%        para.nite is the number of iterations (default: 1000) 
%        para.vstart is the starting value (default: unitary matrix 
%               which diagonalizes first PCA-component of all matrices 
%              regarded as vectors.)  
%     
% Output: v:     demixing matrix, such that v'*M(:,:,f)*v 
%                is approximately diagonal
%         diags: N-K matrix of diagonals for N ISA-components 
%                and for K 'frequencies' f. 
%         parout: right now it contains only a field named err which
%                 contains the costfunction as function of iteration  


% written by Guido Nolte
% please, do not distribute at thus stage.
% 
% The code was designed to diagonalize antisymmetric 
% matrices as outlined in 
% Nolte G, Meinecke FC, Ziehe A, Muller KR. 
% "Identifying interactions in mixed and noisy complex systems."
% Phys Rev E Stat Nonlin Soft Matter Phys. 2006 May;73(5 Pt 1):051913. Epub 2006 May 23. 


nite=1000;
v=[];
itekont=0;
minderivative=1e-7; 
symm=0;
if nargin>1 
  if isfield(para,'nite');
     nite=para.nite;
  end  
  if isfield(para,'backweight');
     backweight=para.backweight;
  end
  if isfield(para,'vstart');
     v=para.vstart;
  end
  if isfield(para,'itekont');
     itekont=para.itekont;
  end  
  if isfield(para,'minderivative');
     minderivative=para.minderivative;
  end  
  if isfield(para,'symm');
     symm=para.symm;
  end
end

if length(v)==0
  v=M2start_loc(M);
end
  
derivative=[];
err=[];
[nchan,nchan,nf]=size(M);
dd=diagerr(M,eye(nchan));
err=[err dd];
erreye=log(dd)
dd=diagerr(M,v);
err=[err dd];
errstart=log(dd)

kont1=0;
Mnorm=0;
 for f=1:nf
    Mnorm=Mnorm+norm(M(:,:,f),'fro')^2; 
 end


alpha=1/Mnorm;
for k=1:nite;

  c1=zeros(nchan);noM=0;
  for f=1:nf
    Mx=M(:,:,f); 
    Mx=M(:,:,f);
    Mx1=Mx*v; Mx2=v'*Mx'*v;Mx2=Mx2-diag(diag(Mx2));  
    c1=c1+Mx1*Mx2;
    if symm==0;
      Mx3=Mx'*v; Mx4=v'*Mx*v;Mx4=Mx4-diag(diag(Mx4)); 
      c1=c1+Mx3*Mx4;
    end
  end

  c2=v'*c1-2*dd/nchan*eye(nchan); %for natural gradient 
  %c2=c1*v'-2*dd/nchan*eye(nchan); %for natural gradient from left
  %c2=c1-2*inv(v')*dd/nchan*eye(nchan); %for gradient 


  c1norm=norm(c1,'fro');
  c2norm=norm(c2,'fro');

  if k==1; c2old=c2; end
  vnew=v-alpha*(v*c2); %natural gradient   
  %vnew=v-alpha*(c2*v); %natural gradient from left 
  %vnew=v-alpha*c2*v'*v;   
  %vnew=v-alpha*c2; %gradient 


  ddnew=diagerr(M,vnew);

 
 %disp([ 2*alpha*c2norm^2 dd-ddnew])
if k==1
   relderi=c2norm/Mnorm;
end

  if ddnew<dd
    % alpha=alpha*1.5; 
    alpha=alpha*3;  
    dd=ddnew;
    v=vnew;
    c2old=c2;
    relderi=c2norm/Mnorm;
  else
    kont1=0;
    %alpha=alpha/10;  
    alpha=alpha/10;
  end
  err=[err dd];



    derivative=[derivative relderi];
         if c2norm/Mnorm<minderivative
           break
        end
  disp([k log(dd) log(relderi)]);
  if itekont==1 &  round(k/100)*100==k 
    % disp([k/100 ,log(dd),log(alpha)/log(2)/1000,log(c2norm/c1norm)/log(10)])
  end

end


%disp([k/1000,log(dd),log(alpha)/log(2)/1000,log(c1norm),log(c2norm)])
parout.err=err;
parout.derivative=derivative;
diags=zeros(nchan,nf);
for f=1:nf
  diags(:,f)=diag(v'*M(:,:,f)*v);
end

[v, diags]=sort_isa(v,diags);

parout.term1=v'*c1;
parout.term2=2*dd/nchan*eye(nchan);



return;


function vstart=M2start_loc(M);
  [nchan,nchan,nf]=size(M);
  MM=reshape(M,nchan^2,nf);
  a=MM'*MM;
  [u,s,v]=svd(a);
  M1=reshape(MM*u(:,1),nchan,nchan);
  [u,s]=eig(M1);
  vstart=u;
return; 


function [err1,err2,err]=diagerr(M,v);

[nchan,nchan,nf]=size(M);

dv=det(v);
v=v/(dv^(1/nchan));



err1=0;err2=0;
for f=1:nf;
 Mloc=v'*M(:,:,f)*v;
 err1=err1+norm(Mloc-diag(diag(Mloc)),'fro')^2;

 Mloc=M(:,:,f);
 err2=err2+norm(Mloc-diag(diag(Mloc)),'fro')^2;

end


err=err1/err2;

return



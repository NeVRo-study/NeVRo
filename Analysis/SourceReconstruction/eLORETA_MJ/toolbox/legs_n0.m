function [leg0,gradleg]=legs_n0(x,dir,n,type)
% usage: [basis,gradbasis]=legs(x,dir,n,scale)
%
% returns the values and directional derivatives  of (n+1)^2-1 basis functions 
% constructed from spherical harmonics at locations given in x and, for the 
% gradients, for (in general non-normalized) directions given in dir.   
% 
% input: x      set of N locations given as an Nx3 matrix 
%        dir    set of N direction vectors given as an Nx3 matrix 
%                  (dir is not normalized (it hence can be a dipole moment))
%        n       order of spherical harmonics 
%
% output: basis: Nx((n+1)^2-1)  matrix containing in the j.th  row the real 
%                and imaginary parts of r^kY_{kl}(theta,Phi)/(N_{kl}*scale^k) ( (r,theta,phi) 
%                are the spherical coordinates corresponding to  the j.th row in x) 
%                for k=1 to n and l=0 to k 
%                the order is:
%                          real parts for k=1 and l=0,1 (2 terms) 
%                  then    imaginary parts for k=1 and l=1 (1 term) 
%                  then    real parts for k=2 and l=0,1,2 (3 terms) 
%                  then    imaginary parts for k=2 and l=1,2 (2 term) 
%                              etc.
%                   the spherical harmonics are normalized with
%                   N_{kl}=sqrt(4pi (k+l)!/((k-l)!(2k+1)))
%                    the phase does not contain the usual (-1)^l term !!! 
%                   scale is constant preferably set to the avererage radius                   
%
%         gradbasis: Nx((n+1)^2-1) matrix containing in the j.th row the scalar 
%                     product of the gradient of the former with the j.th row of dir
%             
% CC Guido Nolte 
%

[n1,n2]=size(x);

comi=sqrt(-1);

normalize=ones(n,1);
for i=1:n
    normalize(i,1)=i*sqrt(2/(2*i+1));
 end
normalize=repmat(normalize,1,n1);
%normalize3=reshape(repmat(normalize,1,3*n1),3,n,n1);
facto=ones(2*n+2,1);for i=3:2*n+2,facto(i)=facto(i-1)*(i-1);end;


rad=sqrt(x(:,1).^2+x(:,2).^2+x(:,3).^2);
phi=angle(x(:,1)+comi*x(:,2));
costheta=x(:,3)./(rad+eps);
sintheta=sqrt(1-costheta.^2);


ns=1:n;

sintheta_n=(repmat(sintheta,1,n).^repmat(ns,n1,1))';
rad_n=(repmat(rad,1,n).^repmat(ns,n1,1))';

pn1(1,:)=sintheta';
pn0(1,:)=costheta';

if n>1
  pn1(2,:)=costheta'.*3.*sintheta';
  pn0(2,:)=(costheta'.*3.*pn0(1,:)-1)/2;
end
for j=3:n;
       pn0(j,:)=(costheta'.*(-1+2*j).*pn0(j-1,:)-(-1+j)*pn0(j-2,:))/j;
end
for j=2:n-1;
  pn1(1+j,:)=(costheta'.*(1+2*j).*pn1(j,:)-(1+j)*pn1(j-1,:))/j;
end

frn0=pn0.*rad_n;
fin0=0*frn0;
frn1=pn1.*repmat(cos(phi)',n,1).*rad_n;
fin1=pn1.*repmat(sin(phi)',n,1).*rad_n;



% fabrn0(1,3,:)=reshape(ones(1,n1),1,1,n1);
% fabin0(1,3,:)=reshape(zeros(1,n1),1,1,n1);
% for i=2:n
%   fabrn0(i,3,:)=i*frn0(i-1,:);
%   fabin0(i,3,:)=i*fin0(i-1,:);
% end
%   
% for i=2:n
%   fabrn0(i,1,:)=-frn1(i-1,:);
%   fabin0(i,1,:)=0;
%   fabrn0(i,2,:)=-fin1(i-1,:);
%   fabin0(i,2,:)=0.;
% end
% 

fabrn0=reshape(zeros(3,n*n1),3,n,n1);
 fabrn0(3,1,:)=reshape(ones(1,n1),1,1,n1);
 for i=2:n
   fabrn0(3,i,:)=i*frn0(i-1,:);
end
   
for i=2:n
  fabrn0(1,i,:)=-frn1(i-1,:);
  fabrn0(2,i,:)=-fin1(i-1,:);
end



%  
 leg0=frn0;
 
 
 gradleg=zeros(n,n1);
 for i=1:n1
     gradleg(:,i)=(dir(i,:)*fabrn0(:,:,i))';
 end
 
 if type==1
    rad_n=(repmat(rad,1,n).^repmat(-(2*ns+1),n1,1))';
    leg0=leg0.*rad_n;
end

return;

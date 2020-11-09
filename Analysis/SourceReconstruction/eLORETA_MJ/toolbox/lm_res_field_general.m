
function [res,f_theo]=lm_res_field_general(a,f,forwpar,lambda);
F=fcnchk(forwpar.method);


 [npar,ndum]=size(a); 
 if npar==1
     a=a';
     npar=ndum;
 end
 ndip=npar/6; 
 sa=size(a);
 dipall=reshape(a,6,ndip)';
 field=feval(F,dipall,forwpar);  
 [nchan,ndum]=size(field);
 if ndip > 1; 
     f_theo=(sum(field'))';
 else
     f_theo=field;    
 end;
 
 c_theo=zeros(nchan,nchan);
 res=norm(f_theo-f)^2;
 if nargin == 3
     res=res+lambda*norm(dipall(:,4:6),'fro')^2;
 end

  
return;



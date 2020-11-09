function cs_norm=cs2csnorm(cs,cs2);
  [nchan,nchan,nf]=size(cs);
 cs_norm=cs;
  for f=1:nf;
     csloc=cs(:,:,f);
     if nargin==1
       xn=norm(csloc,'fro');
     else
       xn=norm(cs2(:,:,f),'fro');
     end
     cs_norm(:,:,f)=csloc/xn;
   end

return;

     

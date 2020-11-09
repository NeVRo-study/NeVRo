function [dips,c]=para2res(para,ndip,xtype);

  [n,m]=size(para);
   if m==1
       m=n;
   end
  if nargin<3
      xtype='imag';
  end
  
  c=zeros(ndip,ndip);
  
  switch(xtype)
     case{'imag'}
        nc=ndip*(ndip-1)/2;
        
        count=1;
        for i=1:ndip
            for j=i+1:ndip
                c(i,j)=sqrt(-1)*para(count);
                c(j,i)=-c(i,j);
                count=count+1;
            end
        end
        
        
     case{'real'}
        nc=ndip*(ndip+1)/2;
        
        count=1;
        for i=1:ndip
            for j=i:ndip
                c(i,j)=para(count);
                c(j,i)=c(i,j);
                count=count+1;
            end
        end

        
     case{'comp'}
        nc=ndip^2;
        
        count=1;
        for i=1:ndip
            for j=i:ndip
                c(i,j)=para(count);
                c(j,i)=c(i,j);
                count=count+1;
            end
        end
        for i=1:ndip
            for j=i+1:ndip
                c(i,j)=c(i,j)+sqrt(-1)*para(count);
                c(j,i)=c(j,i)-sqrt(-1)*para(count);
                count=count+1;
            end
        end

        
    case{'fixed'}
        nc=1;
        c=[];
  end
  
          dip_par=reshape(para(nc+1:m),5,ndip)';

  theta=dip_par(:,4);
  phi=dip_par(:,5);
  ori=[sin(theta).*cos(phi),sin(theta).*sin(phi),cos(theta)];
  dip_par=[dip_par(:,1:3),ori];
  dips=dip_par;
  
  
    
return
    
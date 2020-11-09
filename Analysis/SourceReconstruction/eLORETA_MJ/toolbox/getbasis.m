function   [bas,gradbas]=getbasis(x1,n1,para,i); 
  scale=100; 
  order=para.order;
  [n,ndum]=size(x1);
  x1=x1-repmat(para.center,n,1);
  if i==1
     [bas,gradbas]=legs(x1,n1,order,scale); 
%      bas0=1./sqrt(x1(:,1).^2+x1(:,2).^2+x1(:,3).^2);
%      gradbas0=(n1(:,1).*x1(:,1)+n1(:,2).*x1(:,2)+n1(:,3).*x1(:,3))./((x1(:,1).^2+x1(:,2).^2+x1(:,3).^2).^(3/2));
%      bas=[bas,bas0];
%      gradbas=[gradbas,gradbas0];
 elseif i==2
     [bas,gradbas]=legs_b(x1,n1,order,scale); 
 elseif i==3
      [bas,gradbas]=legs_c(x1,n1,order,scale); 
 end
 
 return


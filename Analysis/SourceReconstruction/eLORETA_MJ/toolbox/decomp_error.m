function [err, inds]=decomp_error(v1,v2);

[nchan, n1]=size(v1);
[nchan, n2]=size(v2);


errs=zeros(n2,n1);
for i=1:n1
    for j=1:n2
        c=v1(:,i)'*v2(:,j)/norm(v1(:,i))/norm(v2(:,j));
        %errs(j,i)=sqrt(1-c^2);
        errs(j,i)=1-abs(c);
        
    end
end

maxerrs=max(max(errs));

res=zeros(n1,3);
if n1>1
  errsloc=errs;
  for i=1:n1
    [emin, jminvec]=min(errsloc);
    [eminmin imin]=min(emin);
    jmin=jminvec(imin);
    errsloc(jmin,:)=2*maxerrs;
    errsloc(:,imin)=2*maxerrs;
    res(i,:)=[eminmin imin jmin];
  end
else
  [emin, jmin]=min(errs);
  res(i,:)=[eminm 1 jmin];
end
  err=mean(res(:,1));
  inds=res(:,2:3);
  
return;




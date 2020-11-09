function elecs=line21020(x,vals)

[np ndum]=size(x);

y=x(2:end,:)-x(1:end-1,:);
d= sqrt(sum(y.^2,2));
d=[0;d];
d=cumsum(d);
d=d/d(end);



nelec=length(vals);

elecs=zeros(nelec,3);
for i=1:nelec;
    [dmin imin]=min((d-vals(i)).^2);
    elecs(i,:)=x(imin,:);
end




return;
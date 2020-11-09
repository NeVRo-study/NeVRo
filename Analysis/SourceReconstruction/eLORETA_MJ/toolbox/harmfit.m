function [coeffs,err]=harmfit(vc,center,order);
% fits spherical harmonics to a set 
% of surface points 

vc=vc(:,1:3);
[nsurf,ndum]=size(vc);
vc=vc-repmat(center,nsurf,1);
basis=legs_ori(vc,order);

rad=sqrt(vc(:,1).^2+vc(:,2).^2+vc(:,3).^2);

coeffs=inv(basis'*basis)*(basis'*rad);

err=sqrt(mean(abs(rad-basis*coeffs).^2));

return

function [coeffs,err]=harmfit(vc,center,order);
% fits spherical harmonics to a set 
% of surface points 

vc=vc(:,1:3);
[nsurf,ndum]=size(vc);
vc=vc-repmat(center,nsurf,1);
rad=sqrt(vc(:,1).^2+vc(:,2).^2+vc(:,3).^2);
nblock=ceil(nsurf/100);
for i=1:nblock;
    disp(i);
    if i<nblock 
        vcloc=vc((i-1)*100+1:i*100,:);
        radloc=rad((i-1)*100+1:i*100);
    else
        vcloc=vc((i-1)*100+1:end,:);
        radloc=rad((i-1)*100+1:end);
    end
    
    basis=legs_ori(vcloc,order);
    if i==1
        c=basis'*basis;
        r=basis'*radloc;
    else
        c=c+basis'*basis;
        r=r+basis'*radloc;
    end
end

    


coeffs=inv(c)*(r);

modelvc=mk_vcharm(vc,center,coeffs);
vc=vc-repmat(center,nsurf,1);
modelrad=sqrt(sum(modelvc.^2,2));
err=sqrt(mean(abs(rad-modelrad).^2));

return

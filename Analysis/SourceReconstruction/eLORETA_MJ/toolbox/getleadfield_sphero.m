
function lead0=getleadfield_sphero(x1,n1,para);


if isfield(para,'senslocs')
   senslocs=para.senslocs;
   sensus=para.uall;
   sensds=para.dall;
   [nsurf,ndum]=size(x1);
   [nsens,ndum]=size(senslocs);
   lead0=getlead(sensus,sensds,senslocs,x1,n1);
else
   lead0=0;
   nsens=1;
end

if isfield(para,'refloc')
    refloc=para.refloc;
    refu=para.refu;
    refd=para.refd;
    leadref=getlead(refu,refd,refloc,x1,n1);
    lead0=lead0-repmat(leadref,nsens,1);
end

return

function lead0=getlead(sensus,sensds,senslocs,x1,n1);

[nsurf,ndum]=size(x1);
[nsens,ndum]=size(senslocs);
lead0=zeros(nsens,nsurf);


for i=1:nsens
    u=reshape(sensus(i,:,:),3,3);
    curv=sensds(i,:);
    abar=(curv(1)+curv(2))/2;
    dela=(curv(1)-curv(2))/2;
    sensloc=senslocs(i,1:3)';
    for j=1:nsurf;
        xloc=x1(j,:)';
        xdir=u'*n1(j,:)';
        d=u'*(xloc-sensloc);
        nd=norm(d);
        if nd>sqrt(eps)
          rz=nd-d(3);
          gradrz=d/nd-[0;0;1];
          finf=-d'*xdir/nd^3;
          fspher=abar*xdir'*gradrz/(nd-d(3));
          fsphero=-dela*xdir'*( (d(1)*[1;0;0]-d(2)*[0;1;0])/rz^2-(d(1)^2-d(2)^2)/rz^3*gradrz);
           lead0(i,j)=2*(finf-fspher-fsphero);
           %lead0(i,j)=1/nd;
        else
            lead0(i,j)=0;
        end
    end
end


return


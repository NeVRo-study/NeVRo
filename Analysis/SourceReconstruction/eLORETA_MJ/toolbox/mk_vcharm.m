function vcout=mk_vcharm(vc,center,coeffs);
% returns something 

[nbasis,ndum]=size(coeffs);
order=sqrt(nbasis)-1;

vc=vc(:,1:3);
[nsurf,ndum]=size(vc);
vc=vc-repmat(center,nsurf,1);

block=1000;
nblock=ceil(nsurf/block);

vcoutall=[];
k=1;
for i=1:nblock
    kb=min(k+block-1,nsurf);
    blockloc=kb-k+1;
    vcloc=vc(k:kb,:);
    k=k+block;
    
ori=repmat([1,0,0],blockloc,1);[basis,gradbasis_x]=legs_ori_grad(vcloc,ori,order);
rads=basis*coeffs;
rads0=sqrt(vcloc(:,1).^2+vcloc(:,2).^2+vcloc(:,3).^2);
fakt=repmat(rads./rads0,1,3);

vcout=vcloc.*fakt;
ori=repmat([1,0,0],blockloc,1);[basis,gradbasis_x]=legs_ori_grad(vcout,ori,order);
ori=repmat([0,1,0],blockloc,1);[basis,gradbasis_y]=legs_ori_grad(vcout,ori,order);
ori=repmat([0,0,1],blockloc,1);[basis,gradbasis_z]=legs_ori_grad(vcout,ori,order);



normals=vcloc./repmat(rads0,1,3)-[gradbasis_x*coeffs,gradbasis_y*coeffs,gradbasis_z*coeffs];
normnormals=sqrt(normals(:,1).^2+normals(:,2).^2+normals(:,3).^2);
normals=normals./repmat(normnormals,1,3);
vcout=vcloc.*fakt+repmat(center,blockloc,1);
vcout=[vcout,normals];
vcoutall=[vcoutall;vcout];
[np,ndum]=size(vcoutall);
disp(['number of points so far: ',num2str(np)]);
end

vcout=vcoutall;

return;


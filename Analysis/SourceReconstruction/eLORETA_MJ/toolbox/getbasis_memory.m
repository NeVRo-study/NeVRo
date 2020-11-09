function [bas,a]=getbasis_memory(x,ori,para,type)

 [nsurf,ndum]=size(x);
 
block=1000;
nblock=ceil(nsurf/block);

 a=[];
 bas=[];
k=1;
for i=1:nblock
    kb=min(k+block-1,nsurf);
    blockloc=kb-k+1;
    xloc=x(k:kb,:);
    oriloc=ori(k:kb,:);
    k=k+block;
    [bas_0,a_0]=getbasis(xloc,oriloc,para,type); 
     a=[a;a_0];
     bas=[bas;bas_0];
 end
 
 return;
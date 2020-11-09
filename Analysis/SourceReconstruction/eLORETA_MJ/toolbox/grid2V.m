function V=grid2V(grid,fp);
% makes forward calculation for a grid for unit
% dipoles in all 3 directions
% usage  V=grid2V(grid,fp);
% 
% input: 
%  grid   nx3 matrix, each row is a the location  
%  fp     structure to make forward calculation
% output:
% V   nxmx3 matrix, for m gridpoints
%     V(i,j,k) is the potential of unit dipole at location 
%              grid(j,:) in k.th direction 

[ng,ndum]=size(grid);

dip0=[grid(1,:),[1 0 0]];
v0=forward_general(dip0,fp);
nchan=length(v0);

V=zeros(nchan,ng,3);
E=eye(3);
for i=1:3;
    e0=E(i,:);
    dips=[grid,repmat(e0,ng,1)];
    V(:,:,i)=forward_general(dips,fp);
end

return; 


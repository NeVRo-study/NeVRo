function v=forward_general_memory(dips,fp);
% usage:  v=forward_general_memory(dips,fp);
% makes forward for 'arbitrary' methods for dipoles.
% 
% same as forward_general except that it calculates dipoles 
% (at most) in bunches of thousand to evoid memory overflow 
% for very many dipoles
% 
% input:
% dips Nx6 matrix (the i.th  row contains  dipole locations 
%      (first  3 numbers, in cm) and moment (second 3 numbers, 
%       in nAm)) 
% fp   structure containing all information (apart from the 
%      dipole parameters)  to make a forwad calculation. 
%      The name of the forward method must be specified 
%       in fp.method. (E.g., if the program is myforward.m 
%       then forward_general calls myforward(dips,fp)) 
% 
% output: 
% v     MxN a matrix: the i.th column is the potential/field of the 
%       i.th dipole 
%

F=fcnchk(fp.method);
[nd ndum]=size(dips);
v=feval(F,dips(1,:),fp);  
nchan=length(v);
v=zeros(nchan,nd);

nx=1000;
for i=1:ceil(nd/nx);
    %i=i
    imin=(i-1)*nx+1;
    imax=min([i*nx,nd]);
    v(:,imin:imax)=feval(F,dips(imin:imax,:),fp);  
end

return;

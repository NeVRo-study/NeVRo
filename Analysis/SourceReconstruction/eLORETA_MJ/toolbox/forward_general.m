function v=forward_general(dips,fp);
% usage:  v=forward_general(dips,fp);
% makes forward for 'arbitrary' methods for dipoles 
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
v=feval(F,dips,fp);  

return;

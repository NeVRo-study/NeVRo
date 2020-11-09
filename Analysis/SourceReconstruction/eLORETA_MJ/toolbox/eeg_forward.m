function potential=eeg_forward(dips,forwpar);
% usage: potential=eeg_forward(dips,forwpar);
% makes forward calculation for EEG in realistic volume 
% conductors
% dips is an Nx6 matrix; the first 3 columns denote locations 
%      of N dipoles, and the last three columns the moments (in nAm)
%      distances are assumed to be in cm, then the output is in in muV
% forwpar is a structure calculated from eeg_ini (or eeg_ini_meta)
%          which contains everything needed to do the forward calculation. 
%         if one adds a component forwpart.shell=i then it is assumed that  
%         the dipole is in the i.th shell. Default is forwpar.shell=1 (i.e. inner shell)
%
%  potential is an MxN matrix; the i.th column is the potential in M channels of the i.th dipole 
% 
%  It is always assumed that the last channel given in the initialization is the reference 
%  channel. The potential in this reference channel itself (which is zero for this reference)
%  is not included in potential, and hence potential has one fewer rows than there were 
%  channels. 
%  If you prefer a difference refererencing you can apply the according  linear
% transformation after the calculation of the potential. Specifically, 
% if A is matrix, then typing forwpar.lintrafo=A before using this program
% then the output will be A*potential instead of potential.
% You can also use this to do spatially weighted fitting. 
%

nshells=length(forwpar.para_global_out);
sigma_k=forwpar.para_global_out{nshells}.sigma;

para_sensors_out=forwpar.para_sensors_out;
para_global_out=forwpar.para_global_out;

if isfield(forwpar,'shell')
    shell=forwpar.shell;
else
    shell=1;
end

lead0=calc_lead0_major(dips(:,1:3),dips(:,4:6),para_sensors_out);
lead1_corr=calc_leadcorr_major(dips(:,1:3),dips(:,4:6),para_sensors_out,shell);
lead2_corr=calc_leadcorr_major_global(dips(:,1:3),dips(:,4:6),para_global_out,shell);
potential=(lead0+lead1_corr+lead2_corr)';

potential=10*potential/(4*pi*sigma_k);


if isfield(forwpar,'lintrafo');
    potential=forwpar.lintrafo*potential;
end


return;

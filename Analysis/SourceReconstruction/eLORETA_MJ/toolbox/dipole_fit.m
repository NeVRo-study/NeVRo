function [a,res_out,k,field_theo]=dipole_fit(f,fp,ndip,para); 
% fits an n-dipole model to a given surface potential or magnetic 
% field at one time-point using the Levenberg-Marquadt  method.
%
% usage: [a,res_out,k,field_theo]=dipole_fit(f,fp,ndip,para); 
%
% input:
% f: nx1 vector with n measurements for n-channels
% fp: structure containing all information to do 
%     the forward calculation (usally fp=sa.fp. 
%     The sructure is in general very complicated. 
%     It might to be generated with meg_ini or eeg_ini in 
%     case of meg and  eeg, respectively.) 
% ndip:    number of dipoles. 
% para:    structure containing optional parameters: 
%          para.astart contains initial values 
%             if it is of size 6xndip it contains locations 
%               and moments of ndip dipoles 
%             if it is of size 3xndip it contains just locations 
%               and the moments are fitted. 
%             if para.astart is not given, a random location 
%               is chosen and the moment is fitted. The location 
%               is taken randomly in a sphere of 6cm radius around 
%               the center of the volume conductor. The center location 
%               is contained in fp.
%          para.lambda is a regularization parameter suppressing large 
%          dipole moments. Set para.lambda to a tiny value to prevent 
%          the radial dipole moment to be arbitrary for a spherical 
%          volume conductor in MEG. Default is lambda=0 (means no regularization).
%    
%
% output:  
% a:       ndipx6-matrix containing locations (in cm) and amplitudes (in nAm)  
%          of the solution. 
% res_out: norm of residual, (i.e. the error which is minimized), 
%          divided by norm of original input potential/field
% k : number of iterations performed. 
% field_theo: the forward calculated potential/field of that found solution 
%
%
% Remark: 
% Note that the result is not necessarily identical if you
% apply the program twice, if no initial guess is provided, 
% because the initial location is random.  
% I recommend to apply the program a couple of times and 
% to choose the best fit result in order to 
% avoid local minima. 
%
%

lambda=0; 
astart=ran_dip(fp.centers(1,:),6,ndip,'cart');
if nargin>3
   if isfield(para,'lambda');
      lambda=para.lambda;
   end
   if isfield(para,'astart') 
     astart=para.astart; 
     [np,nd]=size(astart);
     if np==3 
        [astart,res,f_theo]=lm_start_field_general(astart,f,fp,lambda);
     elseif np==6
        astart=astart;
     else
        error('para.astart must be either a 3xndip or a 6xndip matrix')
     end
   end
 end

 [a,res_out,k,field_theo]=lm_field_general(astart,f,fp,lambda);
 
  a=reshape(a', ndip, []);
return; 
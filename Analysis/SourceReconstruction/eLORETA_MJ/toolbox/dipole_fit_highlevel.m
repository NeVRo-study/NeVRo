function [a,res_out,k,field_theo]=dipole_fit_highlevel(sc,f,ndip,par); %

forwpar=sc.fp;

% fits an n-dipole model to a given surface potential or magnetic 
% field at one time-point using the Levenberg-Marquadt  method.
%
% usage: [a,res_out,k,field_theo]=dipole_fit_highlevel(sc,f,ndip,par); 
%
% input:
% f: nx1 vector with n measurements for n-channels
% sc: structure containing all information to do 
%          the forward calculation in sc.fp. The sructure is in general 
%          very complicated. It needs to be generated with 
%          meg_ini or eeg_ini in case of, guess what, meg and 
%          eeg, respectively. 
% ndip:    number of dipoles. 
% par:     strcucture containing optional parameters: 
%          par.astart contains initial values 
%             if it is of size 6xndip it contains locations 
%               and moments of ndip dipoles 
%             if it is of size 3xndip it contains just locations 
%               and the moments are fitted. 
%             if par.astart is not given, a random location 
%               is chosen and the moment is fitted. The location 
%               is taken randomly in sphere of 6cm radius around 
%               the center of the volume conductor. The center location 
%               is contained in forwpar.
%          par.lambda is a regularization parameter suppressing large 
%          dipole moments. Set par.lambda to a tiny value to prevent 
%          the radial dipole moment to be arbitrary for a spherical 
%          volume conductor in MEG. Default is lambda=0 (means no regularization).
%    
%
% output:  
% a:       6xndip-matrix containing locations (in cm) and amplitudes (in nAm)  
%          of the solution. 
% res_out: norm of residual, (i.e. the error which is minimized)
% k : number of iterations performed. 
% field_theo: the forward calculated potential/field of that found solution 
%
%
% Remark: 
% Note that the result is not necessarily identical if 
% apply the program twice, if no initial guess is provided. 
% I recommend to apply the program a couple of times to 
% avoid local minima. 
%
%
lambda=0; 
astart=ran_dip(forwpar.centers(1,:),6,ndip,'cart');
if nargin>3
   if isfield(par,'lambda');
      lambda=par.lambda;
   end
   if isfield(par,'astart') 
     astart=par.astart; 
     [np,nd]=size(astart);
     if np==3 
        [astart,res,f_theo]=lm_start_field_general(astart,f,forwpar,lambda);
     elseif np==6
        astart=astart;
     else
        error('par.astart must be either a 3xndip or a 6xndip matrix')
     end
   end
 end

 [a,res_out,k,field_theo]=lm_field_general(astart,f,forwpar,lambda);
 
return; 
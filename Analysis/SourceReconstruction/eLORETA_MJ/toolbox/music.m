function [s,vmax,imax,dip_mom,dip_loc]=Music(patt,V,grid);
% makes a Music-scan from a a given pattern.    
% 
% usage: [s,vmax,imax,dip_mom,dip_loc]=Music(patt,V,grid);
%
% input: 
%        patt  nxm matrix for n channels; 
%              each column in patt represents a spatial pattern;
%              (only the span(patt) matters; mixing of the patterns has no effect)
%        V     nxmx3 matrix for m grid_points; V(:,i,j) is the potential
%              of a unit dipole at point i in direction j;
%        grid  (optional) mx3 matrix denoting locations of grid points
%               if you omit that the output dip_loc will be empty
%     
%
%  output:
%          s mxk matrix; s(i,k) indicates fit-quality (from 0 (worst) to 1 (best)) at grid-point
%                        i of k.th dipole (i.e. preceeding k-1 dipoles are projected out 
%                        at each location); the first column is the 'usual'
%                        MUSIC-scan
%          imax  number; denotes grid-index of best dipole
%          vmax  nx1 vector; vmax is the field of the best dipole 
%          dip_mom  1x3 vector; dip_mom is the moment of the  best dipole  
%          dip_loc 1x3 vector; dip_loc is the location of the  best dipole
%                              is empty if grid is not provided. 
%
%
data=orth(patt);
[nchan,nx]=size(patt);
[nchan,ng,ndum]=size(V);
nd=min(nx,ndum);
 

[s,vmax,imax]=calc_spacecorr_all(V,data,nd);
  

 dip_mom=vmax2dipmom(V,imax,vmax);
 if nargin>2
   dip_loc=grid(imax,:);
 else
   dip_loc=[];
 end
 
return;

function [s,vmax,imax]=calc_spacecorr_all(V,data,nd)
     [nchan,ng,ndum]=size(V);
      s=zeros(ng,nd);
      for i=1:ng;
         Vortholoc=orth(squeeze(V(:,i,:)));
         s(i,:)=calc_spacecorr(Vortholoc,data,nd);
      end
     [smax,imax]=max(s(:,1));
     Vortholoc=orth(squeeze(V(:,imax,:)));
     vmax=calc_bestdir(Vortholoc,data);
 return
  

function s=calc_spacecorr(Vloc,data_pats,nd)
     A=data_pats'*Vloc;
     [u sx v]=svd(A);
     sd=(diag(sx))';
     s=sd(1:nd);
 return;

function [vmax,s]=calc_bestdir(Vloc,data_pats,proj_pats)
 if nargin==2
     A=data_pats'*Vloc;
     [u s v]=svd(A);
     vmax=Vloc*v(:,1);
     vmax=vmax/norm(vmax); 
     s=s(1,1);
 else
     [n m]=size(Vloc);
     V_proj=orth(Vloc-proj_pats*(proj_pats'*Vloc));
     A=data_pats'*V_proj;
     [u s v]=svd(A);
     BB=(Vloc'*proj_pats);
     Q=inv(eye(m)-BB*BB'+sqrt(eps));
     vmax=Vloc*(Q*Vloc'*(V_proj*v(:,1)));
     vmax=vmax/norm(vmax); 
     s=s(1,1);
  end
return;


function   dips_mom_all=vmax2dipmom(V,imax_all,vmax_all);
  ns=length(imax_all);
  dips_mom_all=zeros(ns,3);
   for i=1:ns
       Vloc=squeeze(V(:,imax_all(i),:));
       v=vmax_all(:,i);
       dip=inv(Vloc'*Vloc)*Vloc'*v;
       dips_mom_all(i,:)=dip'/norm(dip);
   end
   
return;
      
 
          
        


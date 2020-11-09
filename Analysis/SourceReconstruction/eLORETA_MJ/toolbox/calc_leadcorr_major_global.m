function lead=calc_leadcorr_major_global(x,ori,para_global,shell);
% calculates the corrections to the lead field coming from 
% a set of centers, for given type (type=1: singular at infinity,
% type=3: singular at origin, type=2; both 


leadall=[];

para_exp=para_global{shell}.para;
[ndum,ncenter]=size(para_exp);

for i=1:ncenter;
     center=para_exp{i}.center;
     type=para_exp{i}.type;
     order=para_exp{i}.order;
     para_tmp=struct('center',center,'order',order);
     [bas,a]=getbasis_memory(x,ori,para_tmp,type); 
     if i==1
       lead=-a*para_exp{i}.coeffs;
     else
       lead=lead-a*para_exp{i}.coeffs;
     end
end
   

  
return;

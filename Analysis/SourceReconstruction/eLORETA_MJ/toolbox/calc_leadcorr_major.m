function leadall=calc_leadcorr_major(x,ori,para_sensors,shell);
% calculates the corrections to the lead field coming from 
% a set of centers, for given type (type=1: singular at infinity,
% type=3: singular at origin, type=2; both 

[ndum,nsens]=size(para_sensors);

leadall=[];
for k=1:nsens
   para_exp=para_sensors{k}.para_shell{shell}.para;
   [ndum,ncenter]=size(para_exp);

   for i=1:ncenter;
     center=para_exp{i}.center;
     type=para_exp{i}.type;
     order=para_exp{i}.order;
     para_tmp=struct('center',center,'order',order);
     [bas,a]=getbasis(x,ori,para_tmp,type); 
     if i==1
       lead=-a*para_exp{i}.coeffs;
     else
       lead=lead-a*para_exp{i}.coeffs;
     end
   end
   leadall=[leadall,lead];
   
end

   leadall=leadall(:,1:nsens-1)-repmat(leadall(:,nsens),1,nsens-1);

return;

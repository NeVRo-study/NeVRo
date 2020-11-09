function leadall=calc_lead0_major(x,ori,para_sensors);
% calculates the corrections to the lead field coming from 
% a set of centers, for given type (type=1: singular at infinity,
% type=3: singular at origin, type=2; both

leadall=[];
[ndum,nsens]=size(para_sensors);
for i=1:nsens;
  lead0=getleadfield_sphero(x,ori,para_sensors{i})';
  leadall=[leadall,lead0];
end
leadall=leadall(:,1:nsens-1)-repmat(leadall(:,nsens),1,nsens-1);

return;

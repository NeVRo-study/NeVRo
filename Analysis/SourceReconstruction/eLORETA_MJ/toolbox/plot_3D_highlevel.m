function plot3d_highlevel(sc,surf,source,para);

if nargin<3
 source=[];
end
if nargin<4
  para=[];
end

switch lower(surf)
  case 'cortex'
    show_vc_tri(sc.cortex,source,para);
 case 'brain'
    show_vc_tri(sc.vc{1},source,para);
 case 'skull'
    show_vc_tri(sc.vc{2},source,para);
 case 'scalp'
    show_vc_tri(sc.vc{3},source,para);
 otherwise
  error('second argument must be either the string cortex, brain,skull, or scalp');
end

return;

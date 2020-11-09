function show_vc_tri(vc,points,para);
% plots  volume conductor/cortex as surfaces 
% eventually plus sources/dipoles
% usage show_vc_tri(vc,points,para);
% 
%
% vc is a structure with two fields: 
%    vc.tri is Nx3 matrix of integers where each 
%        column denotes the indices of three points 
%        forming a triangle. 
%        (The name can also be vc.tri_coarse but vc.tri is taken first) 
%    The second field is an Mx3 matrix where each row 
%       denotes a location of a point on the shown surface   
%       This second  field can have various names. The program 
%       looks for the point-locations in the name order 
%       'vc_ori_model', 'vc_ori','vc','vc_coarse'
%       (The reason for these names is, that surfaces can slightly 
%       differ depending on the purpose. E.g. for the forward 
%        calculation, the original curry surface ('vc_ori') is 
%       fitted by an analytic model ('vc_ori_model'); 
% 
% points is an optional Kx3 or  Kx6 matrix. 
%   if it is an Kx3 matrix each 
%     row is a location of a point. All points 
%     are shown as red dots. Points within the shown 
%     surface are invisible. 
%   if it is an Kx6 matrix each 
%     row is a location (points(:,1:3)) plus a moment (points(:,4:6)) 
%     of a dipole. All dipoles are shown in BESA style.  
%     Dipoles within the shown surface are visible.  This was  
%     made possible by  moving the location toward to viewer. Therefore 
%     you can't rotate the maps in this mode! To select a 
%     view-direction this must be given as input (see below); Colors   
%     of the dipoles are alternating. All dipoles are normalized to 
%	 unit length (to change that see below).     
%
% para is an optional structure to set details. 
%   If you want set details without showing a source set the 
%   second argument as [] (i.e. an empty matrix) 
% 
% para.normalize =1 (default) normalizes all shown dipoles to init 
%                    length. 0 -> no normalization.  
% para.myviewdir  is a 1X3 vector denoting the direction of the view
%                 (default is para.myviewdir=[1 0 .5], 
%                  i.e. mainly from left and slightly from top)
% 
% para.numsubplots is number of subplots (either 4 or 1 (default)) 
%		   showing 4 views (with fixed orientations) or 
%                  1 view.  
% para.dipcolorstyle (string) ='uni' (default)-> all dipoles are red;
%                     ='mixed' -> dipoles go through different colors
% para.mymarkersize  size of dot for BESA-style dipoles (default is 30)
% para.mylinewidth  linewidth for BESA-style dipoles (default is 2)
%

if nargin<3
    para.normalize=1;
else
    if ~isfield(para,'normalize')
           para.normalize=1;
    end
end

myviewdir=[1 0 .5 ];
numsubplots=1;
mymarkersize=30;
mylinewidth=2;
fsv=0;
dipcolorstyle='uni';
if nargin>2
   if isfield(para,'myviewdir');
     myviewdir=para.myviewdir;
   end   
  if isfield(para,'numsubplots');
     numsubplots=para.numsubplots;
  end  
  if isfield(para,'mymarkersize');
     mymarkersize=para.mymarkersize;
  end  
  if isfield(para,'mylinewidth');
     mylinewidth=para.mylinewidth;
  end
  if isfield(para,'fsv');
     fsv=para.fsv;
  end
 if isfield(para,'dipcolorstyle');
    dipcolorstyle=para.dipcolorstyle;
  end
end


    

if isfield(vc,'vc_ori_model')
    loc=vc.vc_ori_model(:,1:3);
elseif isfield(vc,'vc_ori')
    loc=vc.vc_ori(:,1:3);
elseif isfield(vc,'vc')
    loc=vc.vc(:,1:3);
elseif isfield(vc,'vc_coarse') 
    loc=vc.vc_coarse(:,1:3);
else
    error('first argument structure should be a structure with field vc_ori_model or vc_ori or vc or vc_coarse (list of surface points)')
end

if isfield(vc,'tri');
    tri=vc.tri(:,1:3);
elseif isfield(vc,'tri_coarse')
    disp('using coarse model to show vc');
    loc=vc.vc_coarse(:,1:3);    
    tri=vc.tri_coarse(:,1:3);
else
    error('first argument should be a structure with field named tri or tri_coarse (list of triangles)')
end


if isfield(vc,'faceori');
  para.faceori=vc.faceori;
else
  pp.vc=loc;
  pp.tri=tri;
  %pp=vc2vcnormals(pp); para.faceori=pp.faceori; para.vertexori=pp.vertexori;
end
if isfield(vc,'vertexori');
  para.vertexori=vc.vertexori;
elseif isfield(para,'vertexori')
else  
  pp.vc=loc;
  pp.tri=tri;
  %pp=vc2vcnormals(pp); para.faceori=pp.faceori; para.vertexori=pp.vertexori;
end


figscale=1.1;
mins=min(loc);
maxs=max(loc);
figscalevec=figscale*[mins(1) maxs(1) mins(2) maxs(2) mins(3) maxs(3)];
if fsv==0;
   fsv=1.5*max(abs(figscalevec(1:2:6)-figscalevec(2:2:6)))/2;
end;
loc=loc(:,1:3);


nnargin=nargin;
if nargin >1; 
    [n,m]=size(points);
     if n+m==0;nnargin=1;  m=3;end
   else 
    m=3;
end;

if m==3
   if numsubplots==4

   subplot(2,2,1);    
   view_dir=[1 0 0];
   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   if nnargin>1;
       plot3(points(:,1),points(:,2),points(:,3),'.r','markersize',mymarkersize);
   end
   axis(figscalevec)

   subplot(2,2,2);    
   view_dir=[0 1 0];
   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   if nnargin>1;
     plot3(points(:,1),points(:,2),points(:,3),'.r','markersize',mymarkersize);
   end
   axis(figscalevec)


   h=subplot(2,2,3);    
   view_dir=[0 0 1];
   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   if nnargin>1;
     plot3(points(:,1),points(:,2),points(:,3),'.r','markersize',mymarkersize);
   end
   axis(figscalevec);
   set(h,'ydir','reverse')
  set(h,'xdir','reverse')


   h=subplot(2,2,4);    
   view_dir=[-1 0 0];
   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   if nnargin>1;
     plot3(points(:,1),points(:,2),points(:,3),'.r','markersize',mymarkersize);
   end
   axis(figscalevec);
 
   elseif numsubplots==1

     view_dir=myviewdir;
     showvc_prog_plain(loc,tri,view_dir,para);
     hold on;
     if nnargin>1;
       plot3(points(:,1),points(:,2),points(:,3),'.r','markersize',mymarkersize);
     end
     axis(figscalevec);
    set(gca,'visible','off')
   else
     error('numsubplots must be 1 or 4 (defaults is 4)');
   end

elseif m==6
    
    colors_x={'b','r','g','c','m','y'};w=[1,1,.999];
    for i=1:n
        colors{i}=colors_x{mod(i,6)+1};
    end
    
 
     dipscal=2;

  if numsubplots==4
     subplot(2,2,1);    
     view_dir=[1 0 0];

     min(tri)
     max(tri)


     showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   points_loc=points(:,1:3)+fsv*repmat(view_dir,n,1);
   for i=1:n
       plot3(points_loc(i,1),points_loc(i,2),points_loc(i,3),'.','color',colors{i},'markersize',mymarkersize);
         ori_loc=points(i,4:6);
        ori_loc_norm=ori_loc/norm(ori_loc);
       if para.normalize==1
            pp=[points_loc(i,:);points_loc(i,:)+dipscal*ori_loc_norm];
        else
            pp=[points_loc(i,:);points_loc(i,:)+ori_loc];
        end

       plot3(pp(:,1),pp(:,2),pp(:,3),'color',colors{i},'linewidth',mylinewidth);

   end
   axis(figscalevec)
   axis equal
   
   subplot(2,2,2);    
   view_dir=[0 -1 0];
   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   points_loc=points(:,1:3)+fsv*repmat(view_dir,n,1);
   for i=1:n
       plot3(points_loc(i,1),points_loc(i,2),points_loc(i,3),'.','color',colors{i},'markersize',mymarkersize);
       ori_loc=points(i,4:6);
       ori_loc_norm=ori_loc/norm(ori_loc);
       if para.normalize==1
            pp=[points_loc(i,:);points_loc(i,:)+dipscal*ori_loc_norm];
        else
            pp=[points_loc(i,:);points_loc(i,:)+ori_loc];
        end
            plot3(pp(:,1),pp(:,2),pp(:,3),'color',colors{i},'linewidth',mylinewidth);
   end

   
   axis(figscalevec)
   axis equal
   

   h=subplot(2,2,3);    
   view_dir=[0 0 1];

   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   points_loc=points(:,1:3)+fsv*repmat(view_dir,n,1);
   for i=1:n
       plot3(points_loc(i,1),points_loc(i,2),points_loc(i,3),'.','color',colors{i},'markersize',mymarkersize);
     ori_loc=points(i,4:6);
        ori_loc_norm=ori_loc/norm(ori_loc);
       if para.normalize==1
            pp=[points_loc(i,:);points_loc(i,:)+dipscal*ori_loc_norm];
        else
            pp=[points_loc(i,:);points_loc(i,:)+ori_loc];
        end

   
       plot3(pp(:,1),pp(:,2),pp(:,3),'color',colors{i},'linewidth',mylinewidth);
    
   end
 
   axis(figscalevec)
   axis equal
    set(h,'ydir','reverse')
    set(h,'xdir','reverse')
 
   h=subplot(2,2,4);    
   view_dir=[-1 0 0];

   showvc_prog_plain(loc,tri,view_dir,para);
   hold on;
   points_loc=points(:,1:3)+fsv*repmat(view_dir,n,1);
   for i=1:n
       plot3(points_loc(i,1),points_loc(i,2),points_loc(i,3),'.','color',colors{i},'markersize',mymarkersize);
     ori_loc=points(i,4:6);
        ori_loc_norm=ori_loc/norm(ori_loc);
       if para.normalize==1
            pp=[points_loc(i,:);points_loc(i,:)+dipscal*ori_loc_norm];
        else
            pp=[points_loc(i,:);points_loc(i,:)+ori_loc];
        end

   
       plot3(pp(:,1),pp(:,2),pp(:,3),'color',colors{i},'linewidth',mylinewidth);
    
   end
   figscalevec 
   axis(figscalevec)
   axis equal
   P=get(h,'position')
   %set(h,'position',P.*[1 1.1 1 1.1]);

   elseif numsubplots==1

     view_dir=myviewdir;
  %   showvc_prog_plain(loc,tri,view_dir,para);
     hold on;

   points_loc=points(:,1:3)+fsv*repmat(view_dir,n,1);
   for i=1:n
       if strcmp(dipcolorstyle,'mixed')
          plot3(points_loc(i,1),points_loc(i,2),points_loc(i,3),'.','color',colors{i},'markersize',mymarkersize);
       elseif strcmp(dipcolorstyle,'uni')
          plot3(points_loc(i,1),points_loc(i,2),points_loc(i,3),'.','color','r','markersize',mymarkersize);
       else
         error('dipcolorstyle must be either uni or mixed')
       end

     ori_loc=points(i,4:6);
        ori_loc_norm=ori_loc/norm(ori_loc);
       if para.normalize==1
            pp=[points_loc(i,:);points_loc(i,:)+dipscal*ori_loc_norm];
        else
            pp=[points_loc(i,:);points_loc(i,:)+ori_loc];
       end
    if strcmp(dipcolorstyle,'mixed')
      plot3(pp(:,1),pp(:,2),pp(:,3),'color',colors{i},'linewidth',mylinewidth);
    elseif strcmp(dipcolorstyle,'uni')
     plot3(pp(:,1),pp(:,2),pp(:,3),'color','r','linewidth',mylinewidth);
    else
        error('dipcolorstyle must be either uni or mixed')
    end
   end
    view_dir=myviewdir;
     showvc_prog_plain(loc,tri,view_dir,para);
     hold on;

  
     axis(figscalevec);
   set(gca,'visible','off')
   else
     error('numsubplots must be 1 or 4 (defaults is 4)');
   end


end

   
   
return;

function show_vc_tri(vc,para,varargin);
% usage show_vc_tri(vc,points,para);
% plots dipoles in volume conductor


if nargin<3
    para.normalize=1;
else
    if ~isfield(para,'normalize')
           para.normalize=1;
    end
end

myviewdir=[-1 0 0 ];
numsubplots=4;
mymarkersize=30;
mylinewidth=2;
fsv=0;
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
   fsv=max(abs(figscalevec(1:2:6)-figscalevec(2:2:6)))/2;
end;
fsv=0;
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
   view_dir=myviewdir;
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
   view_dir=myviewdir;

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
    view_dir=myviewdir;
     showvc_prog_plain(loc,tri,view_dir,para);
     hold on;

   xxx=10
     axis(figscalevec);

   else
     error('numsubplots must be 1 or 4 (defaults is 4)');
   end


end

   
   
return;

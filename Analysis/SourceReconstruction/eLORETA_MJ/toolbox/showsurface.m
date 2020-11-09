function showsurface(vc,para,varargin);
% plots  volume conductor/cortex as surfaces 
% eventually plus sources/dipoles
% usage: showsurface(vc,para,source1,source2,....);
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
%
% para is an optional structure to set details. 
%  If you want set details without showing a source set the 
%  second argument as [] (i.e. an empty matrix) 
%  para.normalize =1 (default) normalizes all shown dipoles to init 
%                    length. 0 -> no normalization.  
% para.myviewdir  is a 1X3 vector denoting the direction of the view
%                 (default is para.myviewdir=[1 0 .5], 
%                  i.e. mainly from left and slightly from top)
% 
%
% para.dipcolorstyle (string) ='uni' (default)-> all dipoles are red;
%                     ='mixed' -> dipoles go through different colors
% para.mymarkersize  size of dot for BESA-style dipoles (default is 30)
% para.mylinewidth  linewidth for BESA-style dipoles (default is 2)
% para.dipolecolorstyle ='uni' (default) (dipoles and dots of one source have the
%                       same color); ='mixed' (all dipoles - but not dots-  have different
%                       color.) This is only useful if you have a handful
%                       of dipoles which should have different colors, but
%                       you don't want to pass each dipole as one argument.
% para.mycolormap    sets the colormap as string. Default is mycolormap='hot'     
% para.dotveccolors  sets manually colors of dots and dipoles. 
%                    by, e.g.,
%                    para.dotveccolors{1}='g';para.dotveccolors{2}='k';etc.
%                    
% 
% source1,source2,... are optional Kx3 or  Kx6 matrices, or a Kx1 vector
%     for Kx1, K must coincide with the number of surface nodes 
%        in vc.vc. Right now, you can only show a single Kx1 vector. 
%        The value is shown as color-code on the surface. Nodes with zero
%        value are not shown. 
%   if a source is a Kx3 matrix each 
%     row is a location of a point. 
%   if it is a Kx6 matrix each 
%     row is a location (source(:,1:3)) plus a moment (source(:,4:6)) 
%     of a dipole. All dipoles are shown in BESA style.  
%     Dipoles within the shown surface are visible.  (This was  
%     made possible by  moving the location toward to viewer. Therefore 
%     you can't rotate the maps in this mode! To select a 
%     view-direction this must be given as input (see above)). 
%   Colors of dots and dipoles are identical for all  dots and dipole 
%   within one source (i.e. one argument) but different for different
%    colors. Dipoles are always normalized unless one chooses
%    para.normalize=0 (see above);
% 


if nargin<2
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
dipscal=2;
if nargin>1
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
  if isfield(para,'dipscal');
     dipscal=para.dipscal;
  end
end


    

if isfield(vc,'vc')
    loc=vc.vc(:,1:3);
elseif isfield(vc,'vc_ori')
    loc=vc.vc_ori(:,1:3);
elseif isfield(vc,'vc_ori_model')
    loc=vc.vc_ori_model(:,1:3);
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
fsv=30;




ndum=0;  nval=0;ndotvec=0;
if nargin>2
    nss=length(varargin);
      for k=1:nss
        source_x=varargin{k};
        [ns,ndum]=size(source_x);
         if ndum==3 | ndum==6;
            ndotvec=ndotvec+1;  
            source{ndotvec}=source_x;
         elseif ndum==1;
            nval=nval+1;  
            source_val{nval}=source_x;
            if nval==1;
                para.voxelfield=source_x;
            end
         end
     end
end

colors_x={'b','r','g','c','m','y'};w=[1,1,.999];
for i=1:ndotvec
    colors{i}=colors_x{mod(i-1,6)+1};
end
if strcmp(dipcolorstyle,'mixed');
    npall=zeroth(ndotvec,1)
    for k=1:ndotvec;
        [npall(k),ndum]=size(source{k});
    end
    npmax=max(npall);
    for i=1:npmax
        colors{i}=colors_x{mod(i,6)+1};
    end
end

if isfield(para,'dotveccolors');
    nc=length(para.dotveccolors);
    for i=1:nc
        colors{i}=para.dotveccolors{i};
    end
end


figscale=1.1;
mins=min(loc);
maxs=max(loc);
figscalevec=figscale*[mins(1) maxs(1) mins(2) maxs(2) mins(3) maxs(3)];


showvc_prog_plain(loc,tri,myviewdir,para);
hold on;

for k=1:ndotvec;
    source_x=source{k}; 
    [np,ndum]=size(source_x);
    if ndum==3
       mycolor=colors{k};
       plot3(source_x(:,1),source_x(:,2),source_x(:,3),'.','markersize',mymarkersize,'color',mycolor);
    elseif ndum==6
       for i=1:np;
          if strcmp(dipcolorstyle,'mixed')
              colorloc=colors{i};
          else
              colorloc=colors{k};
          end
          points_loc=source_x(i,1:3)+fsv*myviewdir;
          ori_loc=source_x(i,4:6);
          ori_loc_norm=ori_loc/norm(ori_loc);
          if para.normalize==1
             pp=[points_loc;points_loc+dipscal*ori_loc_norm];
          else
             pp=[points_loc;points_loc+ori_loc];
          end
          plot3(pp(1,1),pp(1,2),pp(1,3),'.','markersize',mymarkersize,'color',colorloc);
          plot3(pp(:,1),pp(:,2),pp(:,3),'color',colorloc,'linewidth',mylinewidth);
       end
    end
end

axis(figscalevec);
set(gca,'visible','off')

   
   
return;

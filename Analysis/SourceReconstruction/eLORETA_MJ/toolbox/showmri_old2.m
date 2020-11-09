function [hh,hii]=showmri(mri,para,source);
% shows mri-slices eventually plus sources 
% usage:  [hh,hii]=showmri(mri,para,source);
% 
% input:
% mri in general complicated structure containing information about 
%      the mri. The necessary fields are mri.data, a variable with 
%      3 indices containing gray value of each voxel. in mri.data(i,j,k)
%      i must run from right to left, j from front to back, and 
%      k from bottom to top. The other necessary field is 
%      mri.scales, a 3x3 matrix with the m.th diagonal denoting the 
%      the physical (in cm) distance between voxels in the m.th direction 
%      Off-diagonals are not used in this version. 
% para an optional structure to set display parameters. 
%      The following is a list of possible fields with the defaults indicated 
%
%      dslice_shown=1;  distance between adjacent slices in centimeters 
%      orientation='sagittal'; (from right) other options are: 'axial' 
%                  (from bottom) and 'coronal' (from front)  
%      mymarkersize=6; (size of dots of sources are shown as points
%      mylinewidth=1;  (linewidth for quiver plots of dipole fields)
%      mydipmarkersize=30; (size of dots for dipoles in BESA-style)
%      mydiplinewidth=4;   (witdh of lines  for dipoles in BESA-style)
%      dipscal=2.;      (length of dipoles in BESA style, if 
% 			 normalized to a unit length) 
%      dipshow='besa';  (style for dipole fields, other option is 'quiver')
%      dipnormalize=1;  (normalized dipoles moments, dipnormalize=0 -> 
%			 no normalization    
%      dipcolorstyle (string) ='uni' (default)-> all dipoles are red;
%                     ='mixed' -> dipoles go through different colors
%      limits_within    (the default should be taken from the mri  structure)
%                       3x2 matrix, the i.th each row indicates the boundaries 
%                       shown in the i.th direction within the  resepctive    
%                       slice. If, e.g. the orientation is sagittal, 
%			only the second and third row matter because 
%		         each slice is a picture in y and z-direction 
%	                matters.   
%      limits_slices    3x2 matrix, the i.th each row indicates the boundaries
%                       in which slices are selected. If, e.g. the 
%			orientaion is sagittal, only the first row 
%	                matters because slices are selected along the x-direction.   
%      colorlimits      1x2 matrix indicating the meaning of upper and 
%			lower limit of the colormap (which is 'hot'). 
%			(Only relevant when the sources are given as 
%                        Kx4 matrix (see below)). The default is 
%			 min  and max of the 4.th column of the sources. 
% source  an Kx3, Kx4 or Kx6 matrix
%         the first 3 columns are always the location. (ith. 
%         row=i.th source).
%         For Kx3 each source is shown as a red dot
%         For KX4 each source is shown as a colored square
%           where the 4.th column represents color       
%         For Kx6 the last 3 columns denote dipole moment.
%           The dipoles are normalized and shown either 
%           in BESA-style (default) or quiver style. 
%           See para-options to change from BESA to quiver 
%           or to show non-normalized dipoles
%         Remark: set para=[], if you use all defaults 
%          and have a 3rd argument (ie. a source to show)
%
% Output:
% hh  list of handles to graphic objects of the sources.
%     This is used for quick updates of the figure. 
% hhi indices of shown sources (sources may fall outside 
%     the shown slices, are not shown and the handles would 
%     get confused)
% 


data=mri.data;
scales=diag(mri.scales);
nnn=size(data);
x0=(0:(nnn(1)-1))*scales(1);
y0=(0:(nnn(2)-1))*scales(2);
z0=(0:(nnn(3)-1))*scales(3);

limits_within=[];
limits_slice=[];

if isfield(mri,'limits_within');
  limits_within=mri.limits_within;
end
if isfield(mri,'limits_slice');
   limits_slice=mri.limits_slice;
end
if isfield(mri,'u_head2mri') &  isfield(mri,'r_head2mri');
   u_head2mri=mri.u_head2mri;  
   r_head2mri=mri.r_head2mri; 
end

if nargin<3
  source=[];
end
if nargin<2
  para=[];
end

ndum=0;
if length(source)>0
  [ns,ndum]=size(source);
  if ndum==3;
     source_locs=(u_head2mri*(source(:,1:3)'+repmat(r_head2mri,1,ns)))';
    source=source_locs;
  elseif ndum==4;   
    source_locs=(u_head2mri*(source(:,1:3)'+repmat(r_head2mri,1,ns)))';
    source_val=source(:,4);
    source_val_max=max(source_val);
    source_val_min=min(source_val);
    colmax=source_val_max;
    colmin=source_val_min;
    source=[source_locs,source_val];
  elseif ndum==6;
    source_locs=(u_head2mri*(source(:,1:3)'+repmat(r_head2mri,1,ns)))';
    source_ori=(u_head2mri*source(:,4:6)')';
    source=[source_locs,source_ori];
  else
     error('second argument must be matrix with either 3 or 6 columns')
  end
end


dslice_shown=1;
orientation='sagittal';
mymarkersize=6;
mylinewidth=1;
mydipmarkersize=20;
mydiplinewidth=2;
dipcolorstyle='uni';
dipscal=2.;
dipshow='besa';
dipnormalize=1;
if nargin>1
  if isfield(para,'limits_within')
    limits_within=para.limits_within; 
  end 
  if isfield(para,'limits_slice')
    limits_slice=para.limits_slice;
  end
  if isfield(para,'dslice_shown');
    dslice_shown=para.dslice_shown;
  end
  if isfield(para,'orientation');
    orientation=para.orientation;;
  end  
  if isfield(para,'mymarkersize');
    mymarkersize=para.mymarkersize;
  end
  if isfield(para,'mylinewidth');
    mylinewidth=para.mylinewidth;
  end
   if isfield(para,'dipshow');
    dipshow=para.dipshow;
   end
 if isfield(para,'dipcolorstyle');
    dipcolorstyle=para.dipcolorstyle;
 end
 if isfield(para,'colorlimits');
    colmin=para.colorlimits(1);
    colmax=para.colorlimits(2);
  end
end


if length(limits_within)==0;
  limits_within=[[x0(1) x0(end)];[y0(1) y0(end)];[z0(1) z0(end)]];
end
if length(limits_slice)==0;
  limits_slice=[[x0(1) x0(end)];[y0(1) y0(end)];[z0(1) z0(end)]];
end

[nx,ny,nz]=size(data);

switch orientation 
   case 'sagittal'
     index=1;
      x=y0; y=z0;   z=x0; 
     loclimits=[limits_within(2,:);limits_within(3,:);limits_slice(1,:)];
   case 'coronal'
     index=2;    
     x=x0; y=z0; z=y0; 
     loclimits=[limits_within(1,:);limits_within(3,:);limits_slice(2,:)];
   case 'axial'
     index=3;    
     x=x0; y=y0; z=z0;  
     loclimits=[limits_within(1,:);limits_within(2,:);limits_slice(3,:)];
   otherwise
     error('orientation must be either coronal, sagittal or axial');
end
    nn=floor((loclimits(3,2)-loclimits(3,1))/dslice_shown+1);
    
  if ~isfield(para,'mymarkersize') & nn==1;
    mymarkersize=12;
  end
    
    dslice=scales(index);

k=0;
if ndum==4;
  ncol=ceil(sqrt(nn+1));
else
  ncol=ceil(sqrt(nn));
end
if length(source)>0 
 [ns,ndum]=size(source);
  allinds=(1:ns)';
end
 
hh=[];
hii=[];
kkk=0;
ncolb=ncol;
if nn==1
    ncol=2;
    ncolb=1;
end
for k=1:nn
 zloc=loclimits(3,1)+(k-1)*dslice_shown;
 iloc=round(zloc/dslice)+1;
  h=subplot(ncolb,ncol,k);
 

  set(h,'drawmode','fast');
  switch index 
     case 1
         dataloc=squeeze(data(iloc,:,:));
      case 2 
       dataloc=squeeze(data(:,iloc,:)); 
       case 3 
       dataloc=squeeze(data(:,:,iloc));  
  end
  dataloc=dataloc'; 
  imagesc(x,y,-dataloc);
  set(gca,'fontweight','bold'); 
  set(gca,'fontsize',12);
  %contourf(x,y,-dataloc);
  %title(num2str(k));
  colormap('gray');
  p=get(h,'position');
  if nn==1
      %p=[p(1) p(2) 1.8*p(3) p(4)*.9];
      p=[p(1)-.05 p(2)+.1 1.3*p(3) p(4)*.7];
  else
      p=[p(1) p(2) 1.2*p(3) p(4)*1.2];
  end
  set(h,'position',p);
  set(gca,'ydir','normal')
  if index==2 | index==3
    set(gca,'xdir','reverse');
  end
  if index==3
    set(gca,'ydir','reverse');
  end
  % axis equal
  axis([loclimits(1,1) loclimits(1,2) loclimits(2,1) loclimits(2,2)]); 
 
  if length(source)>0
    [ns,ndum]=size(source); 
    colors_x={'b','r','g','c','m','y'};w=[1,1,.999];
    for i=1:ns
        colors{i}=colors_x{mod(i,6)+1};
    end

 
    zpos=source(:,index);
    zloc=zloc;
    ii=abs(zpos-zloc)<dslice_shown/2;
     hii=[hii;allinds(ii)];
     source_loc=source(ii,:);
    switch index
      case 1 
        source_pos=source_loc(:,[2,3]);
      case 2 
        source_pos=source_loc(:,[1,3]);
      case 3 
        source_pos=source_loc(:,[1,2]);
    end
    if ndum==6; 
      switch index
	case 1 
          source_ori=source_loc(:,[5,6]);
	case 2 
          source_ori=source_loc(:,[4,6]);
	case 3 
          source_ori=source_loc(:,[4,5]);
      end
    end
    if ndum==4; 
       source_val_loc=source_loc(:,4);
    end
    hold on;
     if length(source_pos)>0 
       if ndum==3
          plot(source_pos(:,1),source_pos(:,2),'r.','markersize',mymarkersize);
       elseif ndum==4;
           [ng,ndum]=size(source_loc); 
           c=hot;
           nc=length(c);
           for i=1:ng 
              icol=ceil((source_val_loc(i,:)-colmin)/(colmax-colmin)*(nc-1)+eps);
              icol=min([icol,nc]);icol=max([icol,1]);
              loccolor=c(icol,:);
              %plot(source_pos(i,1),source_pos(i,2),'.','color',loccolor,'markersize',mymarkersize); 
              h=plot(source_pos(i,1),source_pos(i,2),'s','color',loccolor,'markersize',mymarkersize,'markerfacecolor',loccolor);
              set(h,'erasemode','none'); 
              %set(h,'erasemode','background');
              %set(h,'erasemode','normal');
              hh=[hh;h];
              kkk=kkk+1; 
              %disp([i k kkk h])
            end
           %icol=ceil((source_val_loc(:,:)-colmin)/(colmax-colmin)*(nc-1)+eps); 
	   %loccolor=c(icol,:);  
           %scatter(source_pos(:,1),source_pos(:,2),mymarkersize,loccolor,'filled'); 
       elseif ndum==6;   
     
         if strcmp(dipshow,'quiver');
            hq=quiver(source_pos(:,1),source_pos(:,2),source_ori(:,1),source_ori(:,2),0);
            set(hq(1),'color','r','linewidth',1); 
            if length(hq)>1;
              set(hq(2),'color','r','linewidth',1); 
            end
         elseif   strcmp(dipshow,'besa');  
            [ng,ndum]=size(source_loc);  
            allindsloc=allinds(ii);
            %disp([ng length(allindsloc)]); 
            for i=1:ng 
                if strcmp(dipcolorstyle,'mixed')
                  plot(source_pos(i,1),source_pos(i,2),'.','color',colors{allindsloc(i)},'markersize',mydipmarkersize);
                elseif strcmp(dipcolorstyle,'uni')
                  plot(source_pos(i,1),source_pos(i,2),'.','color','r','markersize',mydipmarkersize);
                else
                  error('dipcolorstyle must be either uni or mixed')
                end
              ori_loc=source_ori(i,:);
              ori_loc_norm=ori_loc/norm(ori_loc);
              if dipnormalize==1
                pp=[source_pos(i,:);source_pos(i,:)+dipscal*ori_loc_norm];
              else
                pp=[source_pos(i,:);source_pos(i,:)+ori_loc];
              end
              if strcmp(dipcolorstyle,'mixed')
                 plot(pp(:,1),pp(:,2),'color',colors{allindsloc(i)},'linewidth',mydiplinewidth);
              elseif strcmp(dipcolorstyle,'uni')
                  plot(pp(:,1),pp(:,2),'color','r','linewidth',mydiplinewidth);
              else
                  error('dipcolorstyle must be either uni or mixed')
              end
            end
          else
             error('para.dipshow must be either quiver or besa (as string)');
         end
       end
    end

  end

   if k<=nn-ncol
      set(gca,'xtick',[]);
    end
   if k~=round((k-1)/ncol)*ncol+1
     set(gca,'ytick',[]);
  end
end


if ndum==4
     subplot(ncolb,ncol,nn+1);
 %colormap hot; 
 caxis('manual')
 y=colmin:(colmax-colmin)/1000:colmax;
 x=[1]; 
 iy=ceil( y*(nc-1)/(colmax-colmin)+eps);
 M=zeros(nc,1,3);
 for i=1:nc
  M(i,1,:)=c(i,:);
  end
  imagesc(x,y,M);
  set(gca,'ydir','normal');
  P=get(gca,'position');
  if nn==1
     PP=[ P(1)+.05 P(2)+.1 .1*P(3) P(4)*.7];
  else
      PP=[ P(1)+.15 P(2)+.01 .2*P(3) P(4)];
  end
  set(gca,'position',PP);
  set(gca,'xtick',[]);
  set(gca,'fontweight','bold');
end



return;


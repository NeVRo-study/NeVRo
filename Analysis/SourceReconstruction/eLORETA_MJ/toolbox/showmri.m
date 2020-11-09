function [hh,hii]=showmri_test(mri,para,varargin);
% shows mri-slices eventually plus sources 
% usage:  [hh,hii]=showmri(mri,para,source1,source2,...);
% 
% input:
% mri in general complicated structure containing information about 
%      the mri. The necessary fields are mri.data, a variable with 
%      3 indices containing gray value of each voxel. in mri.data(i,j,k)
%      i must run from right to left, j from front to back, and 
%      k from bottom to top. The other necessary fields are:
%      1. mri.scales, a 3x3 matrix with the m.th diagonal denoting the 
%      the physical (in cm) distance between voxels in the m.th direction 
%      Off-diagonals are not used in this version.
%      2. mri.u_head2mri
%      (3x3 matrix)  and mri.r_head2mri to transform from head to
%      mri-coordinate system. (If r is 3x1 location in head-system, then 
%       u_head2mri*(r+r_head2mri) is the location in the mri coordinate
%       system.)
%      3. mri.limits_slice (2x3 matrix), the i.th column denotes upper and
%      lower limit (in cm) of the slices shown for the i.th coordinate
%      4. mri.limits_within (2x3 matrix) sets boundaries within each slice; convention 
%       like in mri.limits_slice.
% para an optional structure to set display parameters. 
%      The following is a list of possible fields with the defaults indicated 
%
%      dslice_shown=1;  distance between adjacent slices in centimeters 
%      orientation='sagittal'; (from right) other options are: 'axial' 
%                  (from bottom) 'coronal' (from front) 
%                  and 'all'. In the 'all'-mode three cuts along three directions
%                  are shown. The cuts go through the point specified 
%                   by para.mricenter (default is [0 0 8]);
%      mricenter   1x3 matrix, specifies the 'center' for showing three
%                  slices in three direction (see para.orientation). Only 
%                  relevant if para.orientation='all';
%      centerlinewidth=1  specifies linewidth indicating cuts when setting
%                         para.orientation='all';
%      centerlinecolor='r' sets color of lines indicating cuts when setting
%                         para.orientation='all';
%      showcenterline=1  if set to zero then in the 'all'-mode (see para.orientation) 
%                        the cuts are not shown as lines.
%      mydotmarkersize=6; (size of dots for dotted sources (i.e. 3 columns))
%      mymarkersize=7; (size of squares for valued sources (i.e. 4 columns))
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
%      colorlimits      Mx2 matrix indicating the meaning of upper and 
%			lower limit of the colormap (which is 'hot'). M is the 
%           number of setos of sources with values, i.e. sources with 4 columns. 
%			(Only relevant when the sources are given as 
%                        Kx4 matrix (see below)). The default is 
%			 min  and max of the 4.th column of the sources. 
%      dotveccolor    cell array, indicating the colors for sources
%                      which are either dots (3 columns) or vectors (6
%                      columns). E.g. para.dotveccolor={'r','g','y'} if 
%                      you have three such sets of sources. (One source
%                      input argument corresponds to one set of sources).
%      colormaps       cell array, indicating the colormaps for sources
%                      which are valued (4 columns). E.g. 
%                      para.colormaps={'hot','copper','cool'} r','g','y'} if 
%                      you have three such sets of sources. (One source
%                      input argument corresponds to one set of sources).
%                      As default the colomaps 'hot' 'copper' 'cool' 
%                      are taken and eventually repeated. 
%      dipcolorstyle   setting dipcolorstyle='mixed' gives each dipole 
%                      an indivdual color. In the default
%                      (dipcolorstyle='uni') each set of dipoles has a
%                      separate color.
%     colobars         =1 (default) draws colorbars for valued sources
%                      (i.e. 4 columns). Set colorbars=0 if you don't want
%                      colorbars.
%     myerasemode      'normal' (default), use myerasemode='none' to 
%                       avoid flickering for quick updates. 
%    showaxes           =0 (default) shows no axes, =1 shows axes
%
% source1,source2,...  (optional) Kx3, Kx4 or Kx6 matrices.
%         The first 3 columns are always the location. (ith. 
%         row=i.th source).
%         For Kx3 each source is shown as a red dot ('dotted source')
%         For KX4 each source is shown as a colored square
%           where the 4.th column represents color  ('valued source')     
%         For Kx6 the last 3 columns denote dipole moment. ('vector source')     
%           The dipoles are normalized and shown either 
%           in BESA-style (default) or quiver style. 
%           set para.normalize=0 if you don't want normalized dipoles.
%           See para-options to change from BESA to quiver 
%           or to show non-normalized dipoles
%         Remark: set para=[], if you use all defaults 
%          and have a 3rd argument (ie. a source to show)
%
% Output: Use this only for one set of source!!
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
  nss=0;
end
if nargin<2
  para=[];
end

ndum=0;  nval=0;ndotvec=0;
if nargin>2
    nss=length(varargin);
    ival=0;
    ivals=zeros(nss,1);idotvecs=zeros(nss,1);
    for k=1:nss
        source_x=varargin{k};
        [ns,ndum]=size(source_x);
          if ndum==3;
            source_locs=(u_head2mri*(source_x(:,1:3)'+repmat(r_head2mri,1,ns)))';
            source{k}=source_locs;
            ndotvec=ndotvec+1;
            idotvecs(k)=ndotvec;
          elseif ndum==4;
            ival=ival+1;  
            source_locs=(u_head2mri*(source_x(:,1:3)'+repmat(r_head2mri,1,ns)))';
            source_val=source_x(:,4);
            source_val_max=max(source_val);
            source_val_min=min(source_val);
            colmax{ival}=source_val_max;
            colmin{ival}=source_val_min;
            source{k}=[source_locs,source_val];
            nval=nval+1;
            ivals(k)=nval;
          elseif ndum==6;
            source_locs=(u_head2mri*(source_x(:,1:3)'+repmat(r_head2mri,1,ns)))';
            source_ori=(u_head2mri*source_x(:,4:6)')';
            source{k}=[source_locs,source_ori];
            ndotvec=ndotvec+1;
            idotvecs(k)=ndotvec;
          else
             source{k}=[];
          end
     end
end

for i=1:nval
    ivalm=mod(i-1,3);
    switch ivalm;
       case 0
           cal{i}=hot;
       case 1
           cal{i}=copper;
       case 2
           cal{i}=cool;
     end
end

           
   colors_x={'b','r','g','c','m','y'};w=[1,1,.999];
    for i=1:ndotvec
        colors{i}=colors_x{mod(i-1,6)+1};
    end
     

dslice_shown=1;
orientation='sagittal';
mymarkersize=7;
mydotmarkersize=6;
mylinewidth=1;
mydipmarkersize=30;
mydiplinewidth=2;
dipcolorstyle='uni';
dipscal=2.;
dipshow='besa';
dipnormalize=1;
colorbars=1;
myerasemode='normal';
showaxes=0;
mricenter=[];
centerlinewidth=1;
centerlinecolor='r';
showcenterline=0;
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
    orientation=para.orientation;
    if strcmp(orientation,'all')
        mricenter=[0 0 8];
        showcenterline=1;
    end
  end   
   if isfield(para,'showcenterline');
    showcenterline=para.showcenterline;
   end
   if isfield(para,'centerlinewidth');
    centerlinewidth=para.centerlinewidth;
   end
   if isfield(para,'centerlinecolor');
    centerlinecolor=para.centerlinecolor;
  end
   if isfield(para,'myerasemode');
    myerasemode=para.myerasemode;
  end  
  if isfield(para,'mymarkersize');
    mymarkersize=para.mymarkersize;
  end
  if isfield(para,'mydotmarkersize');
    mydotmarkersize=para.mydotmarkersize;
  end
   if isfield(para,'mydipmarkersize');
    mydipmarkersize=para.mydipmarkersize;
   end
    if isfield(para,'mydiplinewidth');
    mydiplinewidth=para.mydiplinewidth;
  end
  if isfield(para,'mylinewidth');
    mylinewidth=para.mylinewidth;
  end
   if isfield(para,'dipscal');
    dipscal=para.dipscal;
  end
   if isfield(para,'dipshow');
    dipshow=para.dipshow;
   end
   if isfield(para,'showaxes');
    showaxes=para.showaxes;
   end
 if isfield(para,'dipcolorstyle');
    dipcolorstyle=para.dipcolorstyle;
 end
  if isfield(para,'mricenter');
    mricenter=para.mricenter;
 end
 if isfield(para,'colorlimits');
     [nsx,ndum]=size(para.colorlimits);
     if nsx==1;
       for k=1:nval  
         colmin{k}=para.colorlimits(1);
         colmax{k}=para.colorlimits(2);
       end
     elseif nsx==nval;
       for k=1:nval  
         colmin{k}=para.colorlimits(k,1);
         colmax{k}=para.colorlimits(k,2);
       end
     else
       error('para.colorlimits must be a 1x2 or an nx2 matrix, where n is the number of sets of scalared sources');
     end
 end
 if isfield(para,'colormaps');
       for k=1:nval  
         cal{k}=eval(para.colormaps{k});
       end
 end
  if isfield(para,'colorbars');
       colorbars=para.colorbars;
 end
 if isfield(para,'dotveccolor');
   if length(para.dotveccolor)<ndotvec;
       error('para.dotveccolor must have at least as many elements as dotted or vectored sources');
   end
   for i=1:ndotvec
        colors{i}=para.dotveccolor{i};
   end
 end   
end


if length(limits_within)==0;
  limits_within=[[x0(1) x0(end)];[y0(1) y0(end)];[z0(1) z0(end)]];
end
if length(limits_slice)==0;
  limits_slice=[[x0(1) x0(end)];[y0(1) y0(end)];[z0(1) z0(end)]];
end

if length(mricenter)>0
    mricenter=(u_head2mri*(mricenter'+r_head2mri))';
    xm=mricenter(1); ym=mricenter(2); zm=mricenter(3);dm=dslice_shown/2.001*0;
    limits_slice=[[xm-dm xm+dm];[ym-dm ym+dm];[zm-dm zm+dm]];
end

[nx,ny,nz]=size(data);

if strcmp(orientation,'all')
    nkori=3;
else
    nkori=1;
end

alloris{1}='sagittal';alloris{2}='coronal';alloris{3}='axial';

for kori=1:nkori;
    if nkori==3
        orientation=alloris{kori};
    end

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
if  colorbars==1;
       ncol=ceil(sqrt(nn+nval));
else
  ncol=ceil(sqrt(nn));
end
if nss>0 
  for k=1:nss
   [ns,ndum]=size(varargin{k});
  allinds{k}=(1:ns)';
  end
end
 
hh=[];
hii=[];
kkk=0;
ncolb=ncol;
if nn==1 & nval<3
    if colorbars==1;
        ncol=1+nval;
    else
        ncol=1;
    end
    ncolb=1;
end

if colorbars==1 & nkori==1;
    nsubplot=nn+nval;
elseif colorbars==0 & nkori==1;
    nsubplot=nn ;
else
    nsubplot=1000;
end

for k=1:nn
 zloc=loclimits(3,1)+(k-1)*dslice_shown;
 iloc=round(zloc/dslice)+1;
 if nsubplot>1
     if nkori==3
        h=subplot(2,2,kori);
     else 
        h=subplot(ncolb,ncol,k);
     end
 else
     h=gca;
 end
 
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
  %contourf(x,y,-dataloc,50);
  %hx=surface(x,y,0*dataloc,-dataloc);get(hx)
  
  if showcenterline==1
      hold on;
      switch index
          case 1
              cx=mricenter(2);cy=mricenter(3);
          case 2
              cx=mricenter(1);cy=mricenter(3);
          case 3
              cx=mricenter(1);cy=mricenter(2);
      end
      plot([cx;cx],[-1000;+1000],'color',centerlinecolor,'linewidth',centerlinewidth);
      plot([-1000;+1000],[cy,cy],'color',centerlinecolor,'linewidth',centerlinewidth);
  end
  
  h=gca;
  if nn==1;
      h0=h;
  end
  set(h,'drawmode','fast');

  set(gca,'fontweight','bold'); 
  set(gca,'fontsize',12);
  %contourf(x,y,-dataloc);
  %title(num2str(k));
  colormap('gray');
  p=get(h,'position');
  if nn==1 & nval<3
      %p=[p(1) p(2) 1.8*p(3) p(4)*.9];
      px=[p(1)-.05 p(2)+.1 1.3*p(3) p(4)*.7];
      px=[p(1)-.05 p(2)+.001 1.1*p(3) p(4)*1.1];
      set(h,'position',px);
  else
      %p=[p(1) p(2) 1.2*p(3) p(4)*1.2];
      px=[p(1) p(2) p(3)*1.2 p(4)*1.2];
      set(h,'position',px);
  end
  
  set(gca,'ydir','normal')
  if index==2 | index==3
    set(gca,'xdir','reverse');
  end
  if index==3
    set(gca,'ydir','reverse');
  end
  % axis equal
  axis([loclimits(1,1) loclimits(1,2) loclimits(2,1) loclimits(2,2)]); 
 
 
   idotvec=0;
   for kk=1:nss;
      source_x=source{kk};
     [ns,ndum]=size(source_x); 
     %disp([ns,kk,ndum,ivals(kk)])  
     zpos=source_x(:,index);
     %zloc=zloc
     %zpos=zpos
     %whos
     ii=abs(zpos-zloc)<dslice_shown/2;
     %whos
     hii=[hii;allinds{kk}(ii)];
     source_loc=source_x(ii,:);
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
           idotvec=idotvec+1;
           mydotstyle=strcat(colors{idotvecs(kk)},'.');
           plot(source_pos(:,1),source_pos(:,2),mydotstyle,'markersize',mydotmarkersize);
       elseif ndum==4;
           %disp([ivals(kk),kk])
           ival=ivals(kk);
           [ng,ndum]=size(source_loc); 
           c=cal{ival};
                   
           nc=length(c);
           for i=1:ng 
              icol=ceil((source_val_loc(i,:)-colmin{ival})/(colmax{ival}-colmin{ival})*(nc-1)+eps);
              icol=min([icol,nc]);icol=max([icol,1]);
              loccolor=c(icol,:);
              %plot(source_pos(i,1),source_pos(i,2),'.','color',loccolor,'markersize',mymarkersize); 
              h=plot(source_pos(i,1),source_pos(i,2),'s','color',loccolor,'markersize',mymarkersize,'markerfacecolor',loccolor);
              %set(h,'erasemode','none'); 
              %set(h,'erasemode','background');
              %set(h,'erasemode','normal');
              set(h,'erasemode',myerasemode);
              hh=[hh;h];
              kkk=kkk+1; 
              %disp([i k kkk h])
            end
           %icol=ceil((source_val_loc(:,:)-colmin)/(colmax-colmin)*(nc-1)+eps); 
	   %loccolor=c(icol,:);  
           %scatter(source_pos(:,1),source_pos(:,2),mymarkersize,loccolor,'filled'); 
       elseif ndum==6;
         idotvec=idotvec+1;
         if strcmp(dipshow,'quiver');
            hq=quiver(source_pos(:,1),source_pos(:,2),source_ori(:,1),source_ori(:,2),0);
             mydipcol=colors{idotvecs(kk)};
            set(hq(1),'color',mydipcol,'linewidth',1); 
            if length(hq)>1;
              set(hq(2),'color',mydipcol,'linewidth',1); 
            end
         elseif   strcmp(dipshow,'besa');  
            [ng,ndum]=size(source_loc);  
            allindsloc=allinds{kk}(ii);
            %disp([ng length(allindsloc)]); 
            for i=1:ng 
                if strcmp(dipcolorstyle,'mixed')
                  plot(source_pos(i,1),source_pos(i,2),'.','color',colors{allindsloc(i)},'markersize',mydipmarkersize);
                elseif strcmp(dipcolorstyle,'uni')
                  mydipcolor=colors{idotvecs(kk)};
                  plot(source_pos(i,1),source_pos(i,2),'.','color',mydipcolor,'markersize',mydipmarkersize);
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
                  mydipcolor=colors{idotvecs(kk)};
                  plot(pp(:,1),pp(:,2),'color',mydipcolor,'linewidth',mydiplinewidth);
              else
                  error('dipcolorstyle must be either uni or mixed')
              end
            end
         else
             error('para.dipshow must be either quiver or besa (as string)');
         end
       %drawnow;
       end
     end

   end
  
   if showaxes==1;
       if k<=nn-ncol
          set(gca,'xtick',[]);
       end
       if k~=round((k-1)/ncol)*ncol+1
         set(gca,'ytick',[]);
       end
   else
        set(gca,'xtick',[]);
         set(gca,'ytick',[]);
   end
end
end

if nval>0 & colorbars==1;
  
  hold on;
  for kk=1:nval
      c=cal{kk};
      nc=length(c); 
      if nkori==3
          subplot(2,2,4);
      else
%           ncolb=ncolb
%           ncol=ncol
%           nval=nval
%           nn=nn
%           kk=kk
          subplot(ncolb,ncol,nn+kk);
      end
      nval=nval;
      %colormap hot; 
      caxis('manual')
      y=colmin{kk}:(colmax{kk}-colmin{kk})/1000:colmax{kk};
      x=[1]; 
      iy=ceil( y*(nc-1)/(colmax{kk}-colmin{kk})+eps);
      M=zeros(nc,1,3);
      for i=1:nc
        M(i,1,:)=c(i,:);
      end
      imagesc(x,y,M);
      set(gca,'ydir','normal');
      P=get(gca,'position');
      if nkori==3
          %P0=get(h0,'position');
          %P0=[P0(1) P0(2) (1.1+(kk-1)*.2)*P0(3) P0(4)];
          %set(h0,'position',P0);
         PP=[ P(1)+.1+(nval-kk)*.12 P(2)+.01 .2*P(3) P(4)*.9];
      elseif nn==1 & nval<3
          %P0=get(h0,'position');
          %P0=[P0(1) P0(2) (1.1+(kk-1)*.2)*P0(3) P0(4)];
          %set(h0,'position',P0);
         PP=[ P(1)+.1+(nval-kk)*.12 P(2)+.1 .2*P(3) P(4)*.7];
         %drawnow
      elseif nn==1 & nval>2
          %P0=get(h0,'position');
          %P0=[P0(1) P0(2)-.02 P0(3) P0(4)];
          %set(h0,'position',P0);
          PP=[ P(1)+.1 P(2)+.0 .2*P(3) P(4)];
      else
          PP=[ P(1)+.1 P(2)+.0 .2*P(3) P(4)];
      end
      set(gca,'position',PP);
      set(gca,'xtick',[]);
      set(gca,'fontweight','bold');
  end
end



return;


function [hh,hii]=showmri(mri,para,source);

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
mydipmarkersize=22;
mydiplinewidth=2;   
dipscal=2.5;
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
  %contourf(x,y,-dataloc);
  %title(num2str(k));
  colormap('gray');
  p=get(h,'position');
  p=[p(1) p(2) 1.2*p(3) p(4)*1.2];
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
              loccolor=c(icol,:);
              %plot(source_pos(i,1),source_pos(i,2),'.','color',loccolor,'markersize',mymarkersize); 
              h=plot(source_pos(i,1),source_pos(i,2),'s','color',loccolor,'markersize',mymarkersize,'markerfacecolor',loccolor);
              %set(h,'erasemode','none'); 
              set(h,'erasemode','background');
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
            set(hq(2),'color','r','linewidth',1); 
         elseif   strcmp(dipshow,'besa');  
            [ng,ndum]=size(source_loc);  
            allindsloc=allinds(ii);
            disp([ng length(allindsloc)]); 
            for i=1:ng 
              plot(source_pos(i,1),source_pos(i,2),'.','color',colors{allindsloc(i)},'markersize',mydipmarkersize);
              ori_loc=source_ori(i,:);
              ori_loc_norm=ori_loc/norm(ori_loc);
              if dipnormalize==1
                pp=[source_pos(i,:);source_pos(i,:)+dipscal*ori_loc_norm];
              else
                pp=[source_pos(i,:);source_pos(i,:)+ori_loc];
              end
              plot(pp(:,1),pp(:,2),'color',colors{allindsloc(i)},'linewidth',mydiplinewidth);
            end
          else
             error('para.dipshow must be either quiver or besa (as string)');
         end
       end
    end

  end
  
end


if ndum==4
 subplot(ncol,ncol,nn+1);
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
  PP=[ P(1)+.15 P(2)+.01 .2*P(3) P(4)];
  set(gca,'position',PP);
  set(gca,'xtick',[]);
end



return;


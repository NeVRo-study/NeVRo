function show_isa(vi,diags,locs_2D,para);

[nchan,nf]=size(diags);
kmax=12;
kjump=0;
reso=50;
xf=0:nf-1;
nplot=12;
compjump=2;
myaxis=[];
tf=[];
myxlabel=[];
myylabel=[];
if nargin>3
   if isfield(para,'resolution');
      reso=para.resolution;
    end   
    if isfield(para,'kmax');
      kmax=para.kmax;
    end
    if isfield(para,'xf');
      xf=para.xf;
    end
    if isfield(para,'seco');
      seco=para.seco;
    end
   if isfield(para,'kjump');
      kjump=para.kjump;
    end
 if isfield(para,'tf');
      tf=para.tf;
    end
 if isfield(para,'frange');
      frange=para.frange;
      xf=frange(1)+xf/(nf-1)*(frange(2)-frange(1));
    end
 if isfield(para,'nplot');
      nplot=para.nplot;
      kmax=nplot;
    end
 if isfield(para,'compjump');
      compjump=para.compjump;
    end
 if isfield(para,'myaxis');
      myaxis=para.myaxis;
    end
 if isfield(para,'myxlabel');
      myxlabel=para.myxlabel;
    end
 if isfield(para,'myylabel');
      myylabel=para.myylabel;
    end

end


xpara.resolution=reso;

if nplot==5

  k0=0;
  for kk=1+kjump:kmax+kjump;
    k=compjump*(kk-1)+1;
    k0=k0+1;
    pat=vi(:,k);a=real(pat);a=a/norm(a);b=imag(pat);b=b/norm(b);
    for i=1:4;
      subplot(kmax,5,5*(k0-1)+i);
      phi=(i-1)/4*pi;
      p=real(exp(sqrt(-1)*phi)*(a+sqrt(-1)*b));
      xpara.cbar=0;
      showfield_general(p,locs_2D,xpara); 
      po=get(gca,'position');
      po=[po(1)-.05-(i-4)*.02 po(2)-.0  po(3)*1.23 po(4)*1.23];
      set(gca,'position',po);
    end

    subplot(kmax,5,5*(k0-1)+5)
    if length(tf)==0 
      plot(xf,imag(diags(k,:)));
    else
      xx=imag(diags(k,:));xxmax=max(abs(xx));
      xx=reshape(xx,tf(1),tf(2));
      imagesc(xx); caxis([-xxmax xxmax])
      colorbar;
    end 
  end

elseif nplot==12;

 xpara.cbar=0;
 k0=0;
  for kk=1+kjump:kmax+kjump;
    k=compjump*(kk-1)+1;
    k0=k0+1;
    pat=vi(:,k);no=norm(pat);a=real(pat);a=a/norm(a);b=imag(pat);b=b-(a'*b)*a;b/norm(b);
    subplot(6,6,3*(k0-1)+1); 
    showfield_general(a,locs_2D,xpara); 
    po=get(gca,'position');po=[po(1)-.05+2.5*.02 po(2) po(3)*1.2  po(4)*1.2]; 
    set(gca,'position',po);
    text(-.52,.62,strcat('ISA',num2str(kk)),'fontweight','bold','fontsize',8);
    subplot(6,6,3*(k0-1)+2); 
    showfield_general(b,locs_2D,xpara); 
    po=get(gca,'position');po=[po(1)-.03 po(2)-.0  po(3)*1.2  po(4)*1.2]; 
    set(gca,'position',po);
    subplot(6,6,3*(k0-1)+3); 
    if length(tf)==0 
      plot(xf,no^2*imag(diags(k,:)));
    else
      xx=no^2*imag(diags(k,:));xxmax=max(abs(xx));
      xx=reshape(xx,tf(1),tf(2));
      if length(myaxis)==0
         x=0:tf(2)-1;y=0:tf(1)-1;
      else
         x=[myaxis(1):(myaxis(2)-myaxis(1))/10:myaxis(2)]; 
         y=[myaxis(3):(myaxis(4)-myaxis(3))/10:myaxis(4)];
      end
      imagesc(x,y,xx); caxis([-xxmax xxmax]);
      if kk==kmax+kjump
	if length(myxlabel)>0
          xlabel(myxlabel,'fontweight','bold','fontsize',8);
	end
	if length(myylabel)>0
          ylabel(myylabel,'fontweight','bold','fontsize',8);
	end
      end
      set(gca,'ydir','normal');
      set(gca,'fontweight','bold','fontsize',8);
      %colorbar;
    end 
  end
end

return;

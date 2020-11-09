function update_showmri(hh,vals,col_limits,mythresh);

colmin=col_limits(1);
colmax=col_limits(2);

np=length(hh);
c=hot;
nc=length(c);

 icol=ceil((vals-colmin)/(colmax-colmin)*(nc-1)+eps);
 icolb=min([icol';repmat(nc,1,np)]);
 icolc=max([icolb;repmat(1,1,np)])';


for i=1:np;
  %icol=ceil((vals(i,:)-colmin)/(colmax-colmin)*(nc-1)+eps); 
  %loccolor=c(icol,:); 
  %set(hh(i),'color',loccolor,'markerfacecolor',loccolor); 
  %set(hh(i),'color',loccolor);
  icolloc=icolc(i);
   if vals(i)>mythresh ; 
     set(hh(i),'visible','on')
     set(hh(i),'color',c(icolloc,:),'markerfacecolor',c(icolloc,:));
   else
     set(hh(i),'visible','off')
   end
end


return; 


function [lintrafo, clab_electrodes,chaninds,myinds]=clabstandard2clab(clab_act,clab);


%load clab_standard
nchan_act=length(clab_act);
nchan=length(clab);
%lintrafo=zeros(nchan_act,nchan);
%clab_electrodes=clab_act;
%chaninds=zeros(nchan_act,1);
ii=0;
for i=1:nchan_act
   kont=0;
   for j=1:nchan;
     if strcmp(clab{j},clab_act{i})
         ii=ii+1;
       lintrafo(ii,j)=1;
       chaninds(ii)=j;
       myinds(ii)=i;
       %disp([i,j]);
       kont=1;
       clab_electrodes{ii}=clab{j};
     end
   end
   if kont==0  
        warning(' channel %s not known as surface electrode',clab_act{i});
    end
end

return;
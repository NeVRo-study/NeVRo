function [elecs,inds]=getelecs(e_names,clab,elecs_in)

nout=length(e_names);
nin=length(clab);

elecs=zeros(nout,6);
inds=zeros(nout,1);
for i=1:nout;
  yname=e_names{i};
  for j=1:nin;
      xname=clab{j};
      if strcmp(lower(xname),lower(yname));
          elecs(i,:)=elecs_in(j,:);
          inds(i)=j;
      end
  end
end

return  
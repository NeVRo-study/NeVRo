function   [v_out,diags_out]=sort_isa(v,diags);

vi=inv(v)';

[nchan,nf]=size(diags);
no=zeros(nchan,1);
for i=1:nchan;
  pat=vi(:,i);
  d=diags(i,:)';
  no(i)=norm(d)*norm(pat)^2;
end

[no_s,i_s]=sort(no);

vi_out=vi;
diags_out=diags;

for i=1:nchan
  vi_out(:,i)=vi(:,i_s(nchan+1-i));
  diags_out(i,:)=diags(i_s(nchan+1-i),:);
end

v_out=inv(vi_out)';

return;



  


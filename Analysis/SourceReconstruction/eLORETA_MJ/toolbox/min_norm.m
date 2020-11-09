function [source,source_mag]=min_norm(patt,V,para);
% makes a minimum norm estimate of a potential patt;
%
% input: patt nx1 vector where n is the number of channels;
%        V    nxmx3 tensor where m is the number of gridpoints;
%             V(:,i,j) is the potential of unit dipole at the i.th 
%             point in j.th direction;
%        para  optional structure; weight=0 makes no weighting, weight=1 weights the i.th point 
%              inverse propertional max(max(abs(V(:,i,:))))
% output:  source mx3 matrix containing dipole momemts at each point;
%          sourcee_mag mx1 vector, the magnitude of source for each point

xpinv=0;
xweight=0;

if nargin<3
  para=[];
end


if isfield(para,'pinv')
 xpinv=para.pinv;
end
if isfield(para,'weight')
    xweight=para.weight;
end


if xweight==1
    [nchan,ng,ndum]=size(V);
    %ww=zeros(ng,1);
    for i=1:ng;
        %w=norm(squeeze(V(:,i,:)),'fro');
        w=max(max(abs(squeeze(V(:,i,:)))));
        V(:,i,:)=V(:,i,:)/w;
        %ww(i)=w;
    end
end



if xpinv==1 
  [nx,nchan]=size(V);
  source=reshape(V*patt,nx/3,3); 
   source_mag=(sqrt(sum((source').^2)/3))';
else
  [nchan,ng,ndum]=size(V);
  V=reshape(V,nchan,ng*ndum);
  source=reshape(pinv(V)*patt,ng,ndum);
  source_mag=(sqrt(sum((source').^2)/3))';
end

return;


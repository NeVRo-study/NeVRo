function T=prep_v2head(elecs,head,zmin);
% creates a matrix to make a linear interpolatation 
%  from potentials in electrodes to potentials 
%  on head surface
%
% usage T=prep_v2head(elecs,head);
% 
% input: 
%  elecs    Nx3 matrix (3D coordinates of electrodes)  
%  head     Mx3 matrix (3D coordinates of head surface)  
% output:
% T    MxN matrix, if v is a potential on electrodes 
%                  then Tv is the estimated potential on head surface
%

[minmin,maxmin]=maxdis(elecs);

d1=minmin;
d2=maxmin+.000001;


[nh,nd]=size(head);
[nchan,nd]=size(elecs);
dd=zeros(nh,nchan);
for i=1:nchan;
  rloc=repmat(elecs(i,:),nh,1)-head;
  dd(:,i)=sqrt(sum(rloc.^2,2));
end

T=mytheta(1-dd/(d2*1));

T=sparse(T);

Tm=sum(T,2)+100*eps;
T=T./repmat(Tm,1,nchan);
T(T<1e-8)=0;

if nargin>2
  zmin_elecs=min(elecs(:,3));
  T(head(:,3)<zmin_elecs-zmin,:)=0;
end

return


function [minmin,maxmin]=maxdis(x);
    [n,m]=size(x);

    minall=zeros(n,1);



    for i=1:n
        md=1.e8;
        for j=[1:i-1,i+1:n]
            dis=norm(x(i,:)-x(j,:));
            %disp([i,j,dis]);
              if dis<md
                 md=dis;
              end
        end
        minall(i)=md;
    end

   

    minmin=min(minall);
    [maxmin,imax]=max(minall);
return;

function y=mytheta(x);

y=(x+abs(x))./2;

return;

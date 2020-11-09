function A=wnorm(L,grid,elecs,p,q);
% calculates inverse operator according to weighted minimum norm
% usage A=wnorm(L,grid,elecs);
% Input: 
% L:  nchan x nvoxel x 3 tensor of forward leadfield
% grid: nvoxel x 3 matrix of voxel-locations
% elecs: nchan x 3 matrix of electrode-locations
% Output:
% A (3*nvoxel) x nchan matrix for the inverse map. 
%        The first third of rows gives the x-components, etc. 
% 
% The weights of the i.th voxel is given by 
%  w(i)=||L(:,i,:)||^q/d(i)^p;
% with d(i) being the distance of the i.th voxel to the closest 
% electrode. As default sets  q=1 and p=1.5. 
% Set different values for p and q with 
% A=wnorm(L,grid,elecs,p,q);

[nchan ns ndum]=size(L);

if nargin <4;
    p=1.5;
elseif length(p)==0
    p=1.5;
end
if nargin <5;
    q=1;
elseif length(q)==0
    q=1;
end

W1=(sqrt(sum(sum(L.^2,3),1)))';
W1=repmat(W1,3,1);
W2=calcdensweights(grid,elecs);
W2=repmat(W2,3,1);

Wall=(W2.^p).*(W1.^q);
L=reshape(L,nchan,ns*3);
A=calclintraf(L,Wall);

return;


function w=calcdensweights(grid,locs);

[ns ndum]=size(grid);
[nchan ndum]=size(locs);
w=zeros(ns,1);
for i=1:ns
   d=min(sqrt(sum((locs(:,1:3)-repmat(grid(i,1:3),nchan,1)).^2,2)));
    w(i)=1/d;
end

return;
    
function G=calclintraf(L,W);



[nchan ns]=size(L);

if nargin<2
    W=ones(ns,1);
end
B=(sqrt(1./W))';

Lp=L.*repmat(B,nchan,1);

Lp_pinv=Lp'*inv(Lp*Lp');
%size(B)
%size(Lp_pinv)
G=repmat(B',1,nchan).*Lp_pinv;

return;
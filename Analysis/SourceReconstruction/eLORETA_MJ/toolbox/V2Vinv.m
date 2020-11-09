function Vinv=V2Vpinv(V,para);
% calculates 'inversions' of a leadfield-matrix
% under various notions of inversion
%
% usage:  Vpinv=V2Vpinv(V,para);
% 
% input: 
% V  nxmx3 matrix where V(i,j,k) 
%    is the potential in the i.th channel of 
%    a unit dipole at the j.th grid-point in k.th direction 
% para   optional structure specifying details 
% param.method contains the name of the method
%       possibilites are:
%       'min_norm'  (default) minimum L2_norm 
%       'weighted_min_norm' minimum norm weighted 
%                   with strength of dipoles  
%        ... to be continued 
%
% output 
% Vinv matrix of size 3m X n 
%      which is to be applied on a potential/field v
%      Let S=Vinv*v then Sx=reshape(S,m,3) 
%      is (essentially) the dipole moment for 
%      in m gridpoints for 3 dipole directions  
%       

method='min_norm';
    [nchan,ng,ndum]=size(V);

if nargin<2
  para=[];
end

if isfield(para,'method')
  method=para.method;
end

switch method
  case 'min_norm' 
   disp('using  minimum norm')
     Vinv=pinv(reshape(V,nchan,ng*ndum));

  case 'weighted_min_norm' 
    disp('using weighted minimum norm')
    W=repmat(( sqrt(sum(sum(V.^2,1),3))) ,nchan,1);
    for i=1:3
      V(:,:,i)=V(:,:,i)./W;
    end
    Vinv=pinv(reshape(V,nchan,ng*ndum));
  
  case 'beamformer'
      disp('using beamformer')
      if isfield(para,'cov');
         cov=para.cov;
      else
         cov=eye(nchan);
         disp('warning: no covariance matrix given')
         disp(' using identity matrix');
      end
      covi=inv(cov);
      Vinv=zeros(nchan,ng,ndum);
        for i=1:ng;
          vloc=squeeze(V(:,i,:));
          wloc=covi*vloc*inv(vloc'*covi*vloc);
          Vinv(:,i,:)=wloc;
       end   
       Vinv=(reshape(Vinv,nchan,ng*ndum))';

  otherwise 
    error('unknown method defined in para.method')

end
return;


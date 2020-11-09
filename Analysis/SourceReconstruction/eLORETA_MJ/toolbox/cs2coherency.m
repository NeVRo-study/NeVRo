function cohy=cs2coherency(cs);
% calculates coherency from cross-spectra
% it is assumed that the spatial matrix 
% corresponds to the first two indices
% 3rd (or eventually 4th) index 
% usually correspond to frequency
% (and time in an event related design)

ndim=length(size(cs));

if ndim==2
    cohy=cs./sqrt(diag(cs)*diag(cs)');
elseif ndim==3
    [nchan,nchan,ndim3]=size(cs);
    cohy=zeros(nchan,nchan,ndim3);
    for i=1:ndim3
        csx=cs(:,:,i);
        cohy(:,:,i)=csx./sqrt(diag(csx)*diag(csx)');
    end
elseif ndim==4
    [nchan,nchan,ndim3,ndim4]=size(cs);
    cohy=zeros(nchan,nchan,ndim3,ndim4);
    for i=1:ndim3
        for j=1:ndim4
           csx=cs(:,:,i,j);
           cohy(:,:,i,j)=csx./sqrt(diag(csx)*diag(csx)');
       end
    end
else
    error('cs must have either 2,3 or 4 indices');
end

return;
      
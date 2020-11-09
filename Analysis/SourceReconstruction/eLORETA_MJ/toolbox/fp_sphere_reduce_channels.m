function fpout=fp_sphere_reduce_channels(fp,inds)

fpout=fp;
[ndum, nchan]=size(fp.coeffs);nchan=nchan-1;
nchannew=length(inds);
indsb=inds;indsb(nchannew+1)=nchan+1;

fpout.coeffs=fp.coeffs(:,indsb);
fpout.centers=fp.centers(indsb,:);
fpout.rads=fp.rads(indsb,:);
fpout.sigmas=fp.sigmas(indsb,:);
fpout.rotmats=fp.rotmats(indsb,:,:);
fpout.sensors=fp.sensors(indsb,:);
fpout.lintrafo=eye(nchannew);

return;


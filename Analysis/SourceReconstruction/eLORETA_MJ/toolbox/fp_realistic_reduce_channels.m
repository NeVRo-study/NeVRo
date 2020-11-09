function fpout=fp_realistic_reduce_channels(fp,inds)

fpout=fp;
nshell=length(fp.para_global_out);
nchannew=length(inds);
nchan=length(fp.para_sensors_out)-1;

for i=1:nshell;
    fpout.para_global_out{i}.para{1}.coeffs=fp.para_global_out{i}.para{1}.coeffs(:,inds);
end


for i=1:nchannew;
    pso{i}=fp.para_sensors_out{inds(i)};
end
  pso{nchannew+1}=fp.para_sensors_out{nchan+1};
  
fpout.para_sensors_out=pso;
fpout.lintrafo=eye(nchannew);
return;


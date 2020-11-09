load clab_example;
sa=prepare_sourceanalysis(clab_example);
dips=[[3 -3 9 0 0 1];[3 3 10 0 0 1]];
vv=forward_general(dips,sa.fp);
v=vv(:,1)+vv(:,2);
ndip=2; 

for i=1:10;
[a,res_out,k,field_theo]=dipole_fit_field(v,sa.fp,ndip);
disp([i,res_out])
end

xtype='real';ndip=2;
for i=1:10;
[dips_out,c_source,res_out,c_theo]=dipole_fit_cs(v*v',sa.fp,ndip,xtype);
disp([i,res_out])
end


function  [dips,c_source,res_out,c_theo,a,k,astart]=dipole_fit_cs_highlevel(sc,cs_exp,ndip,xtype); 


[dips,c_source,res_out,c_theo,a,k,astart]=lm_comp_general(cs_exp,sc.fp,ndip,xtype);

return;

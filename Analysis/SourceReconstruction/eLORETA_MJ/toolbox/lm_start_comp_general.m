function [a,res,c_theo]=lm_start_comp_general(a_in,c,forwpar,type)
F=fcnchk(forwpar.method);

[n,ndum]=size(a_in);
ndip=n/5;


if type=='real'; 
    nc=ndip*(ndip+1)/2;nc_R=nc;
elseif type=='imag'
    nc=ndip*(ndip-1)/2;nc_I=nc;
elseif type=='comp'
    nc=ndip^2;nc_R=ndip*(ndip+1)/2;nc_I=ndip*(ndip-1)/2;
else
    error('type must be real, imag or comp');
end

dipall=[];
     for i=1:ndip
        dip=a_in((i-1)*5+1:(i-1)*5+3);
        theta=a_in((i-1)*5+4);
        phi=a_in((i-1)*5+5);
        ori=[ sin(theta)*cos(phi); sin(theta)*sin(phi);cos(theta)];
        dipall_loc=[dip',ori'];
        dipall=[dipall;dipall_loc];
    end
    field_f=feval(F,dipall,forwpar);  
 
    A=field_f'*field_f;
 
 if type=='real' | type=='comp'
    
       E=real(c);
       E_f=E*field_f;
       f_E_f=field_f'*E_f;
    
    
       B=zeros(nc_R,1);
       k=0;
       for i=1:ndip
         for j=i:ndip
           k=k+1;
           fak=2;if i==j;fak=1;end;
           B(k)=2*f_E_f(i,j)*fak;
         end
       end
  
    
       F1=zeros(nc_R,nc_R);                                                                                                                                                                                                                                              
       k1=0;
       for i1=1:ndip;for j1=i1:ndip;
          k1=k1+1;
          k2=0;
          for i2=1:ndip;for j2=i2:ndip;
            k2=k2+1;
            fak1=2;if i1==j1;fak1=1;end;fak2=2;if i2==j2;fak2=1;end;
            F1(k1,k2)=fak1*fak2*(A(j1,j2)*A(i1,i2)+A(j1,i2)*A(i1,j2));
       end;end;end;end;

    aR=inv(F1)*B;
end
if type=='imag' | type=='comp'
    
       E=imag(c);
       E_f=E*field_f;
       f_E_f=field_f'*E_f;
    
    
       B=zeros(nc_I,1);
       k=0;
       for i=1:ndip
         for j=i+1:ndip
           k=k+1;
           B(k)=2*f_E_f(i,j);
         end
       end
  
    
       F1=zeros(nc_I,nc_I);                                                                                                                                                                                                                                              
       k1=0;
       for i1=1:ndip;for j1=i1+1:ndip;
          k1=k1+1;
          k2=0;
          for i2=1:ndip;for j2=i2+1:ndip;
            k2=k2+1;
            F1(k1,k2)=2*(A(j1,j2)*A(i1,i2)-A(j1,i2)*A(i1,j2));
       end;end;end;end;

    aI=inv(F1)*B;
  
end

    
if type=='real'
    a=[aR;a_in];
    [res,c_theo]=lm_res_comp_general(a,c,forwpar,ndip,type);
elseif type=='imag'
    a=[aI;a_in];
    [res,c_theo]=lm_res_comp_general(a,c,forwpar,ndip,type);
elseif type=='comp'
    a=[aR;aI;a_in];
    [res,c_theo]=lm_res_comp_general(a,c,forwpar,ndip,type);
end

    
    
    return 


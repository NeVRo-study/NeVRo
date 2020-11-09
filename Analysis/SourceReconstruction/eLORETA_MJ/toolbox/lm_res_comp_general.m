
function [res,c_theo]=lm_res_comp_general(a,c,forwpar,ndip,type);

F=fcnchk(forwpar.method);

[npar,ndum]=size(a);


if type=='real'; 
    nc=ndip*(ndip+1)/2;nc_R=nc;
    ar=a(1:nc_R);
elseif type=='imag'
    nc=ndip*(ndip-1)/2;nc_I=nc;
    ai=a(1:nc_I);
elseif type=='comp'
    nc=ndip^2;nc_R=ndip*(ndip+1)/2;nc_I=ndip*(ndip-1)/2;
    ar=a(1:nc_R);ai=a(1+nc_R:nc_R+nc_I);
else
    error('type must be real, imag or comp');
end
    
   dipall=[];
     for i=1:ndip
        dip=a(nc+(i-1)*5+1:nc+(i-1)*5+3);
        theta=a(nc+(i-1)*5+4);
        phi=a(nc+(i-1)*5+5);
        ori=[ sin(theta)*cos(phi); sin(theta)*sin(phi);cos(theta)];
        dipall_loc=[dip',ori'];
        dipall=[dipall;dipall_loc];
    end
       field=feval(F,dipall,forwpar);  

        [nchan,ndum]=size(field);
     
   if type=='real' | type=='comp'
        c_theo_R=zeros(nchan,nchan);
     
       k=0;
       for i=1:ndip
           for j=i:ndip
           k=k+1;
           %disp([k,a(k),i,j])
           fak=1;if i==j;fak=.5;end;
            c_theo_R=c_theo_R+ar(k)*fak*(field(:,i)*field(:,j)'+field(:,j)*field(:,i)');   
           end
       end
    
       res_R=norm(c_theo_R-real(c),'fro');
   end
   
    if type=='imag' | type=='comp'
       c_theo_I=zeros(nchan,nchan);
     
       k=0;
       for i=1:ndip
           for j=i+1:ndip
           k=k+1;
           %disp([k,a(k),i,j]
           c_theo_I=c_theo_I+ai(k)*(field(:,i)*field(:,j)'-field(:,j)*field(:,i)');   
           end
       end
       res_I=norm(c_theo_I-imag(c),'fro');
   end
   
   if type=='real'
       res=res_R;
       c_theo=c_theo_R;
   elseif type=='imag'
       res=res_I;
       c_theo=sqrt(-1)*c_theo_I;
   elseif type=='comp'
       res=sqrt(res_R^2+res_I^2);
       c_theo=c_theo_R+sqrt(-1)*c_theo_I;
   end
   
       
    
    
    
return;



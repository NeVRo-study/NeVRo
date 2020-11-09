function [dips,c,res_out,c_theo,a,k,astart]=dipole_fit_cs(c,forwpar,ndip,xtype,astart)
% usage: [dips,c,res_out,c_theo,a,k]=dipole_fit_cs(c,forwpar,ndip,xtype,astart)
%fits a dipole model to the measured covariance matrix c 
% (In general c is a complex cross-spectrum.)
% The fit is done with a random initial guess. You can specify an 
% an initial guess by a fivth argument (see below)
% 
% input: c:  measured cross-spectrum
%        fp: structure containing the forward model (calculated in eeg_ini_meta)
%        ndip: number of dipoles
%        xtype  ='real' (fits only real part); 
%                  ='imag'  (fits only imaginary part)
%                  ='comp' fits both
%        astart: (optional argument) if provided these parameters are used as a starting 
%                point. the order in this vector is for N dipoles: 
%                1. N*(N+1)/2 nonredundant real parts (first column, second column starting from 2nd element, 
%                   3rd column starting 3rd element, etc.) (not present if xtype='imag')
%                2. N*(N-1)/2 nonredundant imaginary parts (first column starting from second element, 2nd column 
%                   starting from 3rd element, etc.) (not present if xtype='real')
%                3. location of first dipole (3 numbers)
%                4. angles theta and phi denoting orientation of first dipole (2 numbers) - the normalized 
%                   dipole vector is then [sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)]
%                5. location of second dipole 
%                6. orientation angles of second dipole
%                .
%                .
%                .  location of N.th dipole
%                .  orientation of N.th dipole
%
% output: dips:     gives a list of dipole locations and normalized moments 
%                   the first 3 numbers in each row are the locations and the next 3 numbers are the moments
%         c_source: estimated cross-spectral matrix of sources
%         err:      relative error defined as ||c_theo-c||/||c|| where ||.|| is the Frobenius norm, i.e. euclidian  
%                   norm for matrices                
%         c_theo:   model cross-spectrum   
%         a:        vector of estimated parameters (this information is redundant and is also contained in dips and          
%                   c_source; the format for "a" is useful if one wants to use this as a starting guess for a new 
%                   search. Then enter "a" as fifth argument.   
% 
% CC Guido Nolte
      
% cfak=norm(c);
% cfakfixed=1000000;
% c=c/cfak*cfakfixed;

if nargin<5
  a_in=ran_dip(forwpar.centers(1,:),5,ndip,'sphe');
[astart,res,c_theo]=lm_start_comp_general(a_in,c,forwpar,xtype);
end

a=astart;
[npar,ndum]=size(a);
if npar==1
    a=a';
    [npar,ndum]=size(a);
end


if xtype=='real'
    norm0=norm(real(c),'fro');
elseif xtype=='imag'
     norm0=norm(imag(c),'fro');
 elseif xtype=='comp'
     norm0=norm(c,'fro');
 else
     error('xtype must be real, imag or comp');
 end
 res0=lm_res_comp_general(a,c,forwpar,ndip,xtype);

alpha=25*norm0;
%alpha=5^10;
k=0;
normdela=1000;

if xtype=='real'; 
    nc=ndip*(ndip+1)/2;
elseif xtype=='imag'
    nc=ndip*(ndip-1)/2;
elseif xtype=='comp'
    nc=ndip^2;
else
    error('xtype must be real, imag or comp');
end


ind_loc=[];for i=1:ndip;ind_loc=[ind_loc,nc+(i-1)*5+1:nc+(i-1)*5+3];end;
dis=1000;
dres=1;

res=res0;

condi=1;

while (normdela>.00001 & k<300 & dis>.001 & dres>1.e-10 & condi<1.e15) | k<20 
k=k+1;
 
    [dela,condi]=lm_1step_general(a,c,forwpar,alpha,ndip,xtype);
    %disp([a,dela]);
    a_new=a+dela;
    %disp([a,dela,a_new])

    res_new=lm_res_comp_general(a_new,c,forwpar,ndip,xtype);
    dres=abs(res_new-res)/res0;
    if res_new<res
    a=a_new;
    alpha=alpha/2;
    res=res_new;
    else
    alpha=alpha*20;
    end
  normdela=norm(dela);
  
  if ndip>1
      dis=dip_dis(a,ndip);
  else
      dis=1;
  end
  %disp(dis)
      %disp([det(F+alpha*eye(npar)),alpha^npar]);
  %disp([k,res,normdela,log(alpha)/log(5),a(1:3)']);
  a_loc=a(ind_loc);[xdum,imax]=max(abs(a_loc));
%disp([k,res/norm0*100,log(normdela),log(alpha)/log(5),a_loc(imax),dis,log(dres)]);
%disp([a_new(2:4)'])
%disp([a_new(1),alpha])
%disp([a(1),a(2:4)',a(7:9)']);


end
  %disp([k,res,normdela,log(alpha)/log(5),a(1:3)']);
[res_new,c_theo]=lm_res_comp_general(a,c,forwpar,ndip,xtype);
res_out=res_new/norm0;

[dips,c]=para2res(a,ndip,xtype);
% 
% c_theo=c_theo*cfak/cfakfixed;
% c=c*cfak/cfakfixed;

return 

function [dela,condi]=lm_1step_general(a,c,forwpar,alpha,ndip,xtype)
F=fcnchk(forwpar.method);

[npar,ndum]=size(a);
C_exp=c;


delta=.00000001;
delta2=2*delta;

if xtype=='real'; 
    nc=ndip*(ndip+1)/2;nc_R=nc;
elseif xtype=='imag'
    nc=ndip*(ndip-1)/2;nc_I=nc;
elseif xtype=='comp'
    nc=ndip^2;nc_R=ndip*(ndip+1)/2;nc_I=ndip*(ndip-1)/2;
else
    error('xtype must be real, imag or comp');
end

      
    
    alp_R=zeros(ndip,ndip);alp_I=zeros(ndip,ndip);

    if xtype=='real' | xtype=='comp'
        k=0;
        for i=1:ndip
            for j=i:ndip
                k=k+1;
                alp_R(i,j)=a(k);
            end
        end
        for i=1:ndip
            for j=1:i
                alp_R(i,j)=alp_R(j,i);
            end
        end
   end
   
   if xtype=='imag' | xtype=='comp'
        if xtype=='imag';k=0;end
        if xtype=='comp';k=ndip*(ndip+1)/2;end
        for i=1:ndip
            for j=i+1:ndip
                k=k+1;
                alp_I(i,j)=a(k);
            end
        end
        for i=1:ndip
            for j=1:i
                alp_I(i,j)=-alp_I(j,i);
            end
        end
    end
        
     
        
          dipall=[];
  

     for i=1:ndip
        dip=a(nc+(i-1)*5+1:nc+(i-1)*5+3);
        theta=a(nc+(i-1)*5+4);
        phi=a(nc+(i-1)*5+5);
        ori=[ sin(theta)*cos(phi); sin(theta)*sin(phi);cos(theta)];
        ori_theta=[ cos(theta)*cos(phi); cos(theta)*sin(phi);-sin(theta)];
        ori_phi=[ -sin(theta)*sin(phi); sin(theta)*cos(phi);0];
        dip_x_u=dip+delta*[1;0;0];dip_x_d=dip-delta*[1;0;0];
        dip_y_u=dip+delta*[0;1;0];dip_y_d=dip-delta*[0;1;0];
        dip_z_u=dip+delta*[0;0;1];dip_z_d=dip-delta*[0;0;1];
    
        dipall_loc=[[dip;ori],[dip_x_u;ori],[dip_x_d;ori],[dip_y_u;ori],[dip_y_d;ori],...
                    [dip_z_u;ori],[dip_z_d;ori],[dip;ori_theta],[dip;ori_phi]]';
        dipall=[dipall;dipall_loc];
    end
    
    %disp(dipall)
    field1=feval(F,dipall,forwpar);  
    [nchan,ndum]=size(field1);
    field=zeros(nchan,6*ndip);
    for i=1:ndip
        field(:, (i-1)*6+1)=field1(:,(i-1)*9+1);
        field(:, (i-1)*6+2)=(field1(:,(i-1)*9+2)-field1(:,(i-1)*9+3))/delta2;
        field(:, (i-1)*6+3)=(field1(:,(i-1)*9+4)-field1(:,(i-1)*9+5))/delta2;
        field(:, (i-1)*6+4)=(field1(:,(i-1)*9+6)-field1(:,(i-1)*9+7))/delta2;
        field(:, (i-1)*6+5)=field1(:,(i-1)*9+8);
        field(:, (i-1)*6+6)=field1(:,(i-1)*9+9);
    end

    
    ind_f=1:6:6*ndip;
    ind_d=[];
    for i=1:ndip
        ind_d=[ind_d,2+(i-1)*6:i*6];
    end

    
    field_f=field(:,ind_f);
    field_d=field(:,ind_d);
    
    A=field_f'*field_f;
    C=field_f'*field_d;
    D=field_d'*field_d;
    
    
  if xtype=='real' | xtype=='comp'
      C_theo_R=zeros(nchan,nchan);
      for i=1:ndip
        for j=1:ndip
                C_theo_R=C_theo_R+alp_R(i,j)*field_f(:,i)*field_f(:,j)';
        end
      end
  end
  if xtype=='imag' | xtype=='comp'
      C_theo_I=zeros(nchan,nchan);
      for i=1:ndip
        for j=1:ndip
                C_theo_I=C_theo_I+alp_I(i,j)*field_f(:,i)*field_f(:,j)';
        end
      end
  end
     
  
  if xtype=='real' | xtype=='comp'
    ER=real(C_exp)-C_theo_R;
    
    ER_f=ER*field_f;
    f_ER_f=field_f'*ER_f;
    ER_d=ER*field_d;
    f_ER_d=field_f'*ER_d;
   
    BR_a=zeros(nc_R,1);
    k=0;
    for i=1:ndip
      for j=i:ndip
        k=k+1;
        fak=1;if i==j;fak=.5;end;
        BR_a(k)=2*f_ER_f(i,j)*fak;
      end
    end
    
    BR_al=zeros(npar-nc,1);
    k=0;
    for i=1:ndip
      for j=1:5
         k=k+1;
         for m=1:ndip
             BR_al(k)=BR_al(k)+2*alp_R(i,m)*f_ER_d(m,(i-1)*5+j);
         end
      end
    end
 

 
    F1_R=zeros(nc_R,nc_R);
    k1=0;
    for i1=1:ndip;for j1=i1:ndip;
         k1=k1+1;
         k2=0;
         for i2=1:ndip;for j2=i2:ndip;
         k2=k2+1;
         fak1=1;if i1==j1;fak1=.5;end;fak2=1;if i2==j2;fak2=.5;end;
         F1_R(k1,k2)=2*fak1*fak2*(A(j1,j2)*A(i1,i2)+A(j1,i2)*A(i1,j2));
%          F1(k1,k2)=2*fak*(A(j1,j2)*A(i1,i2)+A(j1,i2)*A(i1,j2));
     end;end;end;end;
    
    F2_R=zeros(nc_R,npar-nc);
       k1=0;
    for i1=1:ndip;for j1=i1:ndip;
         k1=k1+1;
         fak=1;if i1==j1;fak=.5;end;
         k2=0;
         for i2=1:ndip;for j2=1:5;
            k2=k2+1;
            for i3=1:ndip
                      F2_R(k1,k2)=F2_R(k1,k2)+2*fak*alp_R(i2,i3)*(A(j1,i3)*C(i1,(i2-1)*5+j2)+A(i1,i3)*C(j1,(i2-1)*5+j2));
     end;end;end;end;end;

    F3_R=zeros(npar-nc,npar-nc);
    
    k1=0;
      for i1=1:ndip;for j1=1:5;
         k1=k1+1;
         k2=0;
         for i2=1:ndip;for j2=1:5;
            k2=k2+1;
            for i3=1:ndip;for j3=1:ndip
                    %disp([i1,j1,i2,j2,i3,j3,k1,k2])
               F3_R(k1,k2)=F3_R(k1,k2)+2*alp_R(i1,i3)*alp_R(i2,j3)*(A(i3,j3)*D((i2-1)*5+j2,(i1-1)*5+j1)+C(i3,(i2-1)*5+j2)*C(j3,(i1-1)*5+j1));
     end;end;end;end;end;end;

end


if xtype=='imag' | xtype=='comp'
    EI=imag(C_exp)-C_theo_I;
    
    EI_f=EI*field_f;
    f_EI_f=field_f'*EI_f;
    EI_d=EI*field_d;
    f_EI_d=field_f'*EI_d;
   
    BI_a=zeros(nc_I,1);
    k=0;
    for i=1:ndip
      for j=i+1:ndip
        k=k+1;
        BI_a(k)=2*f_EI_f(i,j);
      end
    end
    
    BI_al=zeros(npar-nc,1);
    k=0;
    for i=1:ndip
      for j=1:5
         k=k+1;
         for m=1:ndip
             BI_al(k)=BI_al(k)-2*alp_I(i,m)*f_EI_d(m,(i-1)*5+j);
         end
      end
    end
 

 
    F1_I=zeros(nc_I,nc_I);
    k1=0;
    for i1=1:ndip;for j1=i1+1:ndip;
         k1=k1+1;
         k2=0;
         for i2=1:ndip;for j2=i2+1:ndip;
         k2=k2+1;
          F1_I(k1,k2)=2*(A(j1,j2)*A(i1,i2)-A(j1,i2)*A(i1,j2));
%          F1(k1,k2)=2*fak*(A(j1,j2)*A(i1,i2)+A(j1,i2)*A(i1,j2));
     end;end;end;end;
    
    F2_I=zeros(nc_I,npar-nc);
       k1=0;
    for i1=1:ndip;for j1=i1+1:ndip;
         k1=k1+1;
         k2=0;
         for i2=1:ndip;for j2=1:5;
            k2=k2+1;
            for i3=1:ndip
                      F2_I(k1,k2)=F2_I(k1,k2)+2*alp_I(i2,i3)*(A(j1,i3)*C(i1,(i2-1)*5+j2)-A(i1,i3)*C(j1,(i2-1)*5+j2));
     end;end;end;end;end;

    F3_I=zeros(npar-nc,npar-nc);
    
    k1=0;
      for i1=1:ndip;for j1=1:5;
         k1=k1+1;
         k2=0;
         for i2=1:ndip;for j2=1:5;
            k2=k2+1;
            for i3=1:ndip;for j3=1:ndip
                    %disp([i1,j1,i2,j2,i3,j3,k1,k2])
               F3_I(k1,k2)=F3_I(k1,k2)+2*alp_I(i1,i3)*alp_I(i2,j3)*(A(i3,j3)*D((i2-1)*5+j2,(i1-1)*5+j1)-C(i3,(i2-1)*5+j2)*C(j3,(i1-1)*5+j1));
     end;end;end;end;end;end;

end




if xtype=='real'
    B=[BR_a;BR_al];
    F=[ [F1_R,F2_R];[F2_R',F3_R]];
elseif xtype=='imag'
    B=[BI_a;BI_al];
    F=[ [F1_I,F2_I];[F2_I',F3_I]];
elseif xtype=='comp'
%     B=[BR_a;-BI_a;BR_al-BI_al];
%     F=[ [F1_R,zeros(nc_R,nc_I),F2_R]; [zeros(nc_I,nc_R), -F1_I,-F2_I]; [F2_R',-F2_I',F3_R-F3_I]];
%     
    B=[BR_a;+BI_a;BR_al+BI_al];
    F=[ [F1_R,zeros(nc_R,nc_I),F2_R]; [zeros(nc_I,nc_R), F1_I, F2_I]; [F2_R',F2_I',F3_R+F3_I]];
end

    IF=inv(F+alpha*eye(npar));
    condi=norm(IF)*norm(F+alpha*eye(npar));
    dela=IF*B;


     
    
        
return;

function dis=dip_dis(a,ndip);

[npar,ndum]=size(a);

nc=npar-5*ndip;

a_dip=reshape(a(nc+1:npar),5,ndip);
a_dip_loc=a_dip(1:3,:);

a_dip_ori=zeros(3,ndip);
for i=1:ndip
    theta=a_dip(4,i);
    phi=a_dip(5,i);
    ori_loc=[sin(theta)*cos(phi);sin(theta)*sin(phi);cos(theta)];
    a_dip_ori(:,i)=ori_loc;
end

diss=zeros(ndip*(ndip-1)/2,1);
k=0;
for i=1:ndip
    for j=i+1:ndip
      k=k+1;
      d_loc=norm(a_dip_loc(:,i)-a_dip_loc(:,j));
      d_ori=min([norm(a_dip_ori(:,i)-a_dip_ori(:,j)),norm(a_dip_ori(:,i)+a_dip_ori(:,j))]);
        diss(k)=d_loc+d_ori;
    end
end

dis=min(diss);

return; 







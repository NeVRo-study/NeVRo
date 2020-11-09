function [a,res_out,k,field_theo]=lm_field_general(astart,f,forwpar,lambda)
%[dips,c,res_out,c_theo,a,k,astart]=lm_comp_general(c,forwpar,ndip,xtype,astart)

if nargin < 4
    lambda=0;
end


[npar,ndum]=size(astart);
astart=reshape(astart,npar*ndum,1);
[npar,ndum]=size(astart);
if npar==1
    astart=astart';
    npar=ndum;
end
ndip=npar/6;
a=astart;
alpha=25;
k=0;
normdela=1000;

dis=1000;
dres=1;
norm0=norm(f);
res0=lm_res_field_general(a,f,forwpar,lambda);
res=res0;
while normdela>.00001 & k<100 & dis>.01 & dres>1.e-7 
k=k+1;
      dela=lm_1step_general(a,f,forwpar,alpha,lambda);
    %disp([a,dela]);
    a_new=a+dela;
    %disp([a,dela,a_new])
     res_new=lm_res_field_general(a_new,f,forwpar,lambda);
    dres=abs(res_new-res)/(res0+sqrt(eps));
    if res_new<res
    a=a_new;
    alpha=alpha/5;
    res=res_new;
  else
    alpha=alpha*5;
  end
  normdela=norm(dela);
  
      if ndip> 1;
          dis=dip_dis(a);
      else
          dis=10;
      end
      %disp(dis)
      %disp([det(F+alpha*eye(npar)),alpha^npar]);
  %disp([k,res,normdela,log(alpha)/log(5)]);
%disp([k,sqrt(res)/norm0*100,normdela,log(alpha)/log(5),dis,dres]);
%disp([a(1),a(2:4)',a(7:9)']);

end
  %disp([k,res,normdela,log(alpha)/log(5),a(1:3)']);
[res_new,field_theo]=lm_res_field_general(a,f,forwpar,0);
res_out=sqrt(res_new)/norm0;

return 

function dela=lm_1step_general(a,f,forwpar,alpha,lambda)
Fmeth=fcnchk(forwpar.method);


[npar,ndum]=size(a);
ndip=npar/6;
f_exp=f;


delta=.000001;
delta2=2*delta;

        
        
          dipall=[];
  

     for i=1:ndip
        loc=a((i-1)*6+1:(i-1)*6+3);
        ori=a((i-1)*6+4:(i-1)*6+6);
        dip_x_u=loc+delta*[1;0;0];dip_x_d=loc-delta*[1;0;0];
        dip_y_u=loc+delta*[0;1;0];dip_y_d=loc-delta*[0;1;0];
        dip_z_u=loc+delta*[0;0;1];dip_z_d=loc-delta*[0;0;1];
        
        loc3=repmat(loc,1,3);
       
      
        dipall_loc=[[dip_x_u;ori],[dip_x_d;ori],[dip_y_u;ori],[dip_y_d;ori],...
                    [dip_z_u;ori],[dip_z_d;ori],[loc3;eye(3)]]';
        dipall=[dipall;dipall_loc];
    end
    
    field1=feval(Fmeth,dipall,forwpar);  
    [nchan,ndum]=size(field1);
    field=zeros(nchan,6*ndip);
    for i=1:ndip
        
        field(:, (i-1)*6+1)=(field1(:,(i-1)*9+1)-field1(:,(i-1)*9+2))/delta2;
        field(:, (i-1)*6+2)=(field1(:,(i-1)*9+3)-field1(:,(i-1)*9+4))/delta2;
        field(:, (i-1)*6+3)=(field1(:,(i-1)*9+5)-field1(:,(i-1)*9+6))/delta2;
        
        
        field(:, (i-1)*6+4)=field1(:,(i-1)*9+7);
        field(:, (i-1)*6+5)=field1(:,(i-1)*9+8);
        field(:, (i-1)*6+6)=field1(:,(i-1)*9+9);

    end

    f_theo=zeros(nchan,1);
    for i=1:ndip;
     f_theo=f_theo+field(:,(i-1)*6+4:(i-1)*6+6)*a((i-1)*6+4:(i-1)*6+6);
    end
    
  
    FTE=field'*(f-f_theo);
    
    F=field'*field;
    
    lambdamat=eye(npar);
    for i=1:ndip
        for j=1:3
           lambdamat( (i-1)*6+j,(i-1)*6+j)=0;
        end
    end
    
    
    dela=inv(F+alpha*eye(npar)+lambda*lambdamat)*(FTE-lambda*lambdamat*a);

    
        
return;

function dis=dip_dis(a);

[npar,ndum]=size(a);
ndip=npar/6;

a_dip=reshape(a,6,ndip);
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
if ndip==1
    dis=1000;
end
return; 







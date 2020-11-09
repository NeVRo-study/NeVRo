function [g1, g2,wall,fminmax]=sica(f1, f2);
% decomposes two neural densities according 
% to a minimum overlap criterion;
% 
% usage: [g1, g2,wall]=sica(f1, f2);
% Input;
%  f1,f2: two dipole fields each having dimension Nx3 where N is 
%        the number of voxels; 
%  Output: 
%  g1, g2: decomposed dipole fields with the same dimension as f1,f2
%  wall: 2x2 decmposition matrix matrix, i.e. 
%         g1=wall(1,1)*f1+wall(1,2)*f2;
%         g2=wall(2,1)*f1+wall(2,2)*f2;
%          if f1 and f2 were generated as inverse solutions from 
%           measured potentials p1 and p2 (each being an Mx1 vector for M
%           channels), then the potentials q1 and q2 of the dipole fields g1 and g2
%           can be calculated directly with P=[p1,p2]; Q=[q1,q2]; with the
%           relation: Q=P*wall'; 
%  fminmax 1x2 vector, minimum and maximum of costfunction 
%           
%         
        


c=[[ trace(f1'*f1) trace(f1'*f2)];[ trace(f2'*f1) trace(f2'*f2)]];

[u s v]=svd(c);
%c=c
%s=s
w=u*sqrt(inv(s))*v';

h1=w(1,1)*f1+w(1,2)*f2;
h2=w(2,1)*f1+w(2,2)*f2;

M=sum(h1.*h2,2);
N=sum(h2.^2-h1.^2,2)/2;

a=sum(M.^2);
b=sum(M.*N);
c=sum(N.^2);



xmin1=atan(2*b/(a-c))/2;
fmin1=a*cos(xmin1)^2+2*b*cos(xmin1)*sin(xmin1)+c*sin(xmin1)^2;
if xmin1<0
    xmin2=xmin1+pi/2;
else
    xmin2=xmin1-pi/2;
end
fmin2=a*cos(xmin2)^2+2*b*cos(xmin2)*sin(xmin2)+c*sin(xmin2)^2;
if fmin1<fmin2;
    xmin=xmin1;
    fmin=fmin1;
    fmax=fmin2;
else
    xmin=xmin2;
    fmin=fmin2;
    fmax=fmin1;
end
phi=xmin/2;
g1=cos(phi)*h1+sin(phi)*h2;
g2=-sin(phi)*h1+cos(phi)*h2;

wall=[[cos(phi) sin(phi)];[-sin(phi) cos(phi)]]*w;
fminmax=[fmin,fmax];
%w=w
%phipi=phi/pi
%fmin/fmax
%phiall=-pi/2:.1:pi/2;
%coall=a*cos(phiall*2).^2+2*b*cos(phiall*2).*sin(phiall*2)+c*sin(phiall*2).^2;
%figure;plot(phiall/pi,coall);
return;

function v=eeg_forward_3sphere(x,fp);
% usage: potential=eeg_forward_3sphere(dips,forwpar);
% makes forward calculation for EEG in 3-shell spherical volume 
% conductors
% dips is an Nx6 matrix; the first 3 columns denote locations 
%      of N dipoles, and the last three columns the moments (in nAm)
%      distances are assumed to be in cm, then the output is in muV
% forwpar is a structure calculated from eeg_ini_3sphere (or eeg_ini_meta)

% 
%  It is always assumed that the last channel given in the initialization is the reference 
%  channel. The potential in this reference channel itself (which is zero for this reference)
%  is not included in potential, and hence potential has one fewer rows than there were 
%  channels. 
%  If you prefer a difference refererencing you can apply the according  linear
% transformation after the calculation of the potential. Specifically, 
% if A is matrix, then typing forwpar.lintrafo=A before using this program
% then the output will be A*potential instead of potential.
% You can also use this to do spatially weighted fitting. 
%



sigma_k=fp.sigmas(1,3);

[ns,ndum]=size(fp.centers);
[nd,ndum]=size(x);

if ndum==1
    x=reshape(x,6,nd/6)';
    nd=nd/6;
end

v=zeros(ns,nd);
yall=[];dirall=[];
for i=1:ns
    u=reshape(fp.rotmats(i,:,:),3,3);
    center=repmat(fp.centers(i,:)',1,nd);
    y=u*(x(:,1:3)'-center);
    dir=u*(x(:,4:6))';
    yall=[yall;y'];
    dirall=[dirall;dir'];
end
    [leg1,gradleg1]=legs_n0(yall,dirall,fp.order,0);
  
    fpc=fp.coeffs(1:fp.order,:);
    fpc=repmat(fpc,nd,1);
    fpc=reshape(fpc,fp.order,nd*ns);
    v_tmp=sum(gradleg1.*fpc);
    v=(reshape(v_tmp,nd,ns))';
    
    
refkont=1;
if isfield(fp,'ref')
    if fp.ref==0
        refkont=0;
    end
end

if refkont==1;
    v=v(1:ns-1,:)-repmat(v(ns,:),ns-1,1);
end
v=10*v/(4*pi*sigma_k);


if isfield(fp,'lintrafo');
    v=fp.lintrafo*v;
end


return;


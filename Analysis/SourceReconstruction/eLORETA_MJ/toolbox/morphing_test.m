ncoeffs=20;strength=.001;


load randomhead 
ii=floor((length(coeffs)-1)*rand(ncoeffs,1))+1;
coeffs_ran_a=coeffs;
coeffs_ran_b=coeffs;
center=[0 0 0 ];
coeffs_ran_a(ii)=coeffs(ii).*(1+randn(ncoeffs,1)*strength);
coeffs_ran_b(ii)=coeffs(ii).*(1+randn(ncoeffs,1)*strength);
skin_a=skin;skin_b=skin;

vcup_x_a=mk_vcharm(vcup_b(:,1:3),center,coeffs_ran_a);
vcup_x_b=mk_vcharm(vcup_b(:,1:3),center,coeffs_ran_b);
skin_a.vc(inds,:)=vcup_x_a(:,1:3);
skin_b.vc(inds,:)=vcup_x_b(:,1:3);

skin_out=skin;
figure;
nn=100;
for i=1:nn;
alpha=(i-1)/(nn-1);
skin_out.vc=alpha*skin_a.vc+(1-alpha)*skin_b.vc;
clf;showsurface(skin_out,[]);drawnow;
end


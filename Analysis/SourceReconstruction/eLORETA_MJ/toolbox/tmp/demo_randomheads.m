ncoeffs=20;strength=.001;


load randomhead 
ii=floor((length(coeffs)-1)*rand(ncoeffs,1))+1;
coeffs_ran=coeffs;
center=[0 0 0 ];
coeffs_ran(ii)=coeffs(ii).*(1+randn(ncoeffs,1)*strength);
skin_b=skin;vcup_c=mk_vcharm(vcup_b(:,1:3),center,coeffs_ran);
skin_b.vc(inds,:)=vcup_c(:,1:3);
figure;showsurface(skin_b,[]);


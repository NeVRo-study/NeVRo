dips=[[3 -7 2.5 0 0 1];[-3.5 -7 2.5 0 0 1]]; 
figure;showmri(sa.mri,[],dips);

dip_left_loc=[3 -7 2.5];
dip_right_loc=[-3.5 -7 2.5];
dip_left=[repmat(dip_left_loc,3,1) eye(3)];
dip_right=[repmat(dip_right_loc,3,1) eye(3)];

v_left=forward_general(dip_left,sa.fp);
v_right=forward_general(dip_right,sa.fp);
figure;
for k=1:3
subplot(2,2,k);
showfield_general(v_left(:,k),sa.locs_2D);
end

figure;
for k=1:3
subplot(2,2,k);
showfield_general(v_right(:,k),sa.locs_2D);
end


b_mat = [];%zeros(38,28);

for i =1:38
    b_mat = [b_mat; results.default(i).model.featuremodel.patterns(1,:)];
end
    
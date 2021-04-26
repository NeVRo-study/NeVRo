
verbose = false;
ppath = './Data/ratings/class_bins/nomov/SA/'; % './Data/ratings/continuous/z_scored/nomov/SA/';
files = dir([ppath, '*.txt']);

for subnum=1:length(files)
    rat_df = readmatrix([files(subnum).folder, '\', files(subnum).name]);

    rats = rat_df(:,2);
    rats_bin = ones(1,240);
    split_low = prctile(rats, 100/3);
    split_high = prctile(rats, 2*100/3);


    for i=1:240
        if rats(i) > split_low
            rats_bin(i) = 2;
        end
        if rats(i) > split_high
            rats_bin(i) = 3;
        end
    end
   
    for split_num = 1:2
        switch split_num
            case 1
                split = split_low;
            case 2
                split = split_high;
        end
        if sum(rats == split)
            idx_cand = find(rats == split);
            n_smaller = split_num * length(rats)/3 - sum(rats < split);
            n_higher = length(idx_cand) - n_smaller;
            if verbose
                fprintf('n_high: %i\n', n_higher);
            end
            idx = randsample(idx_cand, n_higher);
            rats_bin(idx) = rats_bin(idx) + 1;
        end
    end
 
    if verbose
        fprintf([files(subnum).name '\n']);
        fprintf('n at lower: %i\n', sum(rats == split_low));
        fprintf('n at higher: %i\n', sum(rats == split_high));
    end
    for class=1:3
        if verbose
            fprintf('n of class %i --- %i\n', i, sum(rats_bin == i))
        end
        assert( sum(rats_bin == class) == 80)
    end
end





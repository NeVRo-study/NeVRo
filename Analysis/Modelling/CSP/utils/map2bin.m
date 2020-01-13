
function y = map2bin(x) 

for i=1:length(x)
    if abs(x(i)-1) < abs(x(i)-2)
        y(i) = 1
    else
        y(i) = 2
    end
end

vals = [];
for i=1:1000
    be = shuffle(a);
    autBE = autocorr(be);
    vals(i) = mean(abs(autA - autBE));
end
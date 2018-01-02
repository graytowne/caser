function [ res ] = neg_sample(y, ns, max_n)
    x = zeros(max_n, 1);
    x(y) = 1;
    ind_neg = find(x == 0);
    res = ind_neg(randperm(length(ind_neg), ns));
end


function [ res ] = neg_sample(y, ns, max_n)
    % Codes for performing negative sampling
    % 
    % ARGS:
    % y        : the rated item, which we don't want to sample as negatives
    % ns       : the number of negatives we want to sample
    % max_n    : the max number of items
    %
    % RETURN:
    % res       : the sampled negatives

    x = zeros(max_n, 1);
    x(y) = 1;
    ind_neg = find(x == 0);
    res = ind_neg(randperm(length(ind_neg), ns));
end


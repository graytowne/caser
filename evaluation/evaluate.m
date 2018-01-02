function [ ap, prec_k, recall_k ] = evaluate( actual, prediction, cutoff )
% Codes for evaluate average precision (AP), precision@k and recall@k
% given actual ranking and predicted ranking
% 
% ARGS:
% actual       : the given actual ranking
% prediction   : the given predicted ranking
% cutoff       : cutoff k for precision@k and recall@k
%
% RETURN:
% ap           : the average precision (not MAP)
% prec_k       : the precision@k
% recall_k     : the recall@k

ap = ap_k(actual, prediction);
prks = pr_k(actual, prediction, cutoff);
prec_k = prks(1,:);
recall_k = prks(2,:);


function scores = ap_k(actual, prediction, k)
if nargin<3
    k=inf;
end

if length(prediction)>k
    prediction = prediction(1:k);
end
score = 0;
num_hits = 0;
for i=1:min(length(prediction), k)
    if sum(actual==prediction(i))>0 && ...
            sum(prediction(1:i-1)==prediction(i))==0
        num_hits = num_hits + 1;
        score = score + num_hits / i;
    end
end
scores = score / min(length(actual), k);


function scores = pr_k(actual, prediction, ks)
scores = zeros(2, length(ks));
for i=1:length(ks)
    k = ks(i);
    if length(prediction) > k
        pred = prediction(1:k);
    else
        pred = prediction;
    end
    num_hit = length(intersect(actual, pred));
    scores(1, i) = num_hit / length(pred); % precision
    scores(2, i) = num_hit / length(actual); % recall
end
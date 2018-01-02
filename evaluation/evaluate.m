function [ map, auc, prec_k, recall_k ] = evaluate( actual, prediction, cutoff )
% rows = size(actual, 1);
% k = [1, 5, 10];
% 
% aps = zeros(rows, 1);
% aucs = zeros(rows, 1);
% precs = zeros(rows, length(k)); % prec@1, prec@5, prec@10
% recalls = zeros(rows, length(k)); % recall@1, recall@5, recall@10
% 
% for i=1:rows
%     ground = actual{i, :};
%     pred = prediction{i, :};
%     % compute ap
%     aps(i) = ap_k(ground, pred);
%     % compute auc
%     aucs(i) = get_auc(ground, pred);
%     % compute precision and recall
%     prks = pr_k(ground, pred, k);
%     precs(i, :) = prks(1, :);
%     recalls(i, :) = prks(2, :);
% end

map = ap_k(actual, prediction);
auc = get_auc(actual, prediction);
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


function score = get_auc(actual, prediction)
n_drops = 0;
n_rele_items = length(intersect(actual, prediction));
n_eval_items = length(prediction) + n_drops;
n_eval_pairs = (n_eval_items - n_rele_items) * n_rele_items;

if n_eval_pairs == 0
    score = 0.5;
    return;
end

n_correct_pairs = 0;
hits = 0;

for i=1:length(prediction)
    pred = prediction(i);
    if isempty(find(actual == pred, 1))   % actual only contains 1 item
        n_correct_pairs = n_correct_pairs + hits;
    else
        hits = hits + 1;
    end
end

n_miss_items = length(setdiff(actual, prediction));
n_correct_pairs = n_correct_pairs + hits * (n_drops - n_miss_items);
score = n_correct_pairs / n_eval_pairs;



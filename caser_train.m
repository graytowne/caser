function caser_train( args )
    addpath MatConvNet/
    addpath data/
    addpath model/
    addpath optim/
    addpath evaluation/
    addpath utils/

    rng(args.seed);
    fprintf(struct2str(args, 'table'));
    %% Setup: load data
    L = args.L;
    T = args.T;
    train_path = args.trainpath;
    test_path = args.testpath;
    [s_train, s_test, vocab_item, vocab_user, ui_train] = load_data(train_path, test_path, L, T);
    
    %% Setup: initialization
    % dataset related params
    n_train = size(s_train, 1);
    n_test = size(s_test, 1);
    
    fprintf('number of training instances: %d\n', n_train);
    
    n_items = length(vocab_item);
    n_users = length(vocab_user);
    args.n_items = n_items; args.n_users = n_users;
    
    % init learning related args
    batch_size = 512;
    n_batch = ceil(n_train / batch_size);
    rate_once = args.rateonce;
    early_stop = args.earlystop;
    n_iter = args.niter;
    n_NS = args.negsample * T;

    % init model
    caser = Caser(args);
   
    %% Training loop
    losses = zeros(n_iter+1,1);
    
    for iter=1:n_iter
        tic;
        rand_inds = randperm(n_train);
        loss = 0;
        for ind_s=1:n_batch
            % use mini batch (didn't make it parallel)
            inds = (ind_s-1)*batch_size+1: ind_s*batch_size;
            if inds(end) > n_train
                inds = (ind_s-1)*batch_size+1: n_train;
            end
            inds = rand_inds(inds);
            
            item_ls = s_train(inds, 1);
            item_is = s_train(inds, 2);
            users = s_train(inds, 3);
            
            caser.zero_grads();
            batch_loss = 0;
            loss_cnt = 0;
            for ni=1:length(inds)
                item_l = item_ls{ni} + 1;
                item_i = item_is{ni};
                user = users{ni};
                % negtive sampling
                items_rated = unique(ui_train{user});
                item_j = neg_sample(items_rated, n_NS, n_items);
                % compute outputs
                inputs.seq = item_l; inputs.targets = item_i; inputs.user = user; inputs.negatives = item_j;
                outputs = caser.forward(inputs);
                oi = outputs.oi; oj = outputs.oj;
                loss_cnt = loss_cnt + length(oi) + length(oj);
                % compute loss
                l = -(sum(log(vl_nnsigmoid(oi))) + sum(log(vl_nnsigmoid(-oj))));
                doi = vl_nnsigmoid(oi) - 1;
                doj = vl_nnsigmoid(oj);
                batch_loss = batch_loss + l;
                % compute gradients
                inputs.doi = doi; inputs.doj = doj;
                caser.backward(inputs);
            end
            % normalize loss and gradient by total counts
            batch_loss = batch_loss / loss_cnt;
            loss = loss + batch_loss;
            caser.normalize_grads(loss_cnt);
            % update parameters            
            caser.updates();
        end
        loss = loss / n_batch;
        losses(iter+1) = loss;
        delta_loss = losses(iter) - loss;
        fprintf('iter: %d\tloss: %.2f\tdelta_loss: %.2f\t\n', iter, loss, delta_loss);toc;
        
        %% Evaluation loop
        every = 5;
        if early_stop == true && mod(iter, every) == 0 
            tic;
            cutoff = [1,5,10];
            aps = zeros(n_test, 1);
            precs = zeros(n_test, length(cutoff));
            recalls = zeros(n_test, length(cutoff));
            cnt = 1;
            for ind=1:n_test
                item_l = s_test{ind,1} + 1;
                item_i = 1:n_items;
                user = s_test{ind,3};
                label = s_test{ind,2};
                % skip example that the target never seen when training
                if label == 0
                    continue;
                end
                % compute outputs
                inputs.seq = item_l; inputs.targets = item_i; inputs.user = user; inputs.negatives = 1;
                outputs = caser.forward(inputs);
                O = outputs.oi;
                [~, pred] = sort(O, 'descend');
                if rate_once
                    items_rated = ui_train{user};
                    pred = setdiff(pred, items_rated, 'stable');
                end
                % compute metrics
                [ ap, prec_k, recall_k ] = evaluate(label, pred, cutoff);
                aps(cnt,1) = ap;
                precs(cnt,:) = prec_k;
                recalls(cnt,:) = recall_k;
                cnt = cnt + 1;
            end
            
            mean_ap = sum(aps) / (cnt-1);
            mean_prec_k = sum(precs) / (cnt-1);
            mean_recall_k = sum(recalls) / (cnt-1);

            fprintf('map %f\n', mean_ap);
            fprintf('prec@1 %f, prec@5 %f, prec@10 %f\n', mean_prec_k(1), mean_prec_k(2), mean_prec_k(3));
            fprintf('recall@1 %f, recall@5 %f, recall@10 %f\n', mean_recall_k(1), mean_recall_k(2), mean_recall_k(3));
            fprintf('\n');toc;
            
            if iter > every
                if results(iter, 8) < results(iter-every, 8)  % early stop if there's no MAP improves
                    break;
                end
            end
        end
    end
end
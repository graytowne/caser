function [s_train, s_test, vocab_item, vocab_user, ui_train] = load_data(train_path, test_path, L, T) 
    % Codes for loading data from local file
    % 
    % ARGS:
    % train_path    : the given path to training file
    % test_path     : the given path to test file
    % L             : the length of sequence
    % T             : number of targets
    %
    % RETURN:
    % s_train       : the resulted training instances 
    % s_test        : the resulted test instances   
    % vocab_item    : the relationship vocabulary between new item id and origin item id
    % vocab_user    : the relationship vocabulary between new user id and origin user id
    % ui_train      : rated item of each user
    
    max_length = L + T;
    train_ratio = L/max_length;

    fid_train = fopen(train_path);
    fid_test = fopen(test_path);

    % train
    data = textscan(fid_train, '%f\t%f\t%f');
    fclose(fid_train);
    train = cell2mat(data);
    clear data;

    vocab_user = unique(train(:, 1));
    vocab_item = unique(train(:, 2));

    s_train = cell(1e+6, 3); % data, label, user
    s_test = cell(1e+6, 3);
    ui_train = cell(length(vocab_user), 1);

    cnt = 1;
    for i=1:length(vocab_user)
        new_user_id = i;
        old_user_id = vocab_user(i);
        ind = train(:, 1) == old_user_id;
        items = train(ind, 2);
        [~, items_] = ismember(items, vocab_item);
        ui_train{new_user_id} = items_;
        if length(items_) <= max_length
            items_new = zeros(max_length,1);
            items_new(max_length - length(items_)+1:end) = items_;
            ind_train = floor(length(items_new) * train_ratio);

            s_train{cnt, 1} = items_new(1:ind_train);
            s_train{cnt, 2} = items_new(ind_train+1:end);
            s_train{cnt, 3} = new_user_id;
            cnt = cnt + 1;

            s_test{new_user_id, 1} = items_new(end-floor(max_length*train_ratio)+1:end);
            s_test{new_user_id, 3} = new_user_id;
        else
            ind_label = max_length:1:length(items_);
            for j=ind_label
                inds = (j-max_length+1) : j;
                ind_ = floor(max_length * train_ratio);
                ind_train = inds(1: ind_);
                ind_test = inds(ind_+1:end);
                s_train{cnt, 1} = items_(ind_train);
                s_train{cnt, 2} = items_(ind_test);
                s_train{cnt, 3} = new_user_id;
                cnt = cnt + 1;
            end

            s_test{new_user_id, 1} = items_(end-floor(max_length*train_ratio)+1:end);
            s_test{new_user_id, 3} = new_user_id;
        end
    end

    % test
    data = textscan(fid_test, '%f\t%f\t%f');
    fclose(fid_test);
    test = cell2mat(data);
    clear data;

    for i=1:length(vocab_user)
        new_user_id = i;
        old_user_id = vocab_user(i);

        ind = test(:, 1) == old_user_id;
        items = test(ind, 2);
        [~, items_] = ismember(items, vocab_item);
        s_test{new_user_id, 2} = items_;
    end

    empty_cells = cellfun(@isempty, s_train);
    empty_rows = empty_cells(:,1) == 1;
    s_train(empty_rows, :) = [];

    empty_cells = cellfun(@isempty, s_test);
    empty_rows = empty_cells(:,1) == 1;
    s_test(empty_rows, :) = [];
end
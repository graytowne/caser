classdef Caser < handle
    properties
        Args
        Params
        Grads
        States
        Temp_Params
    end
    
    methods
        function obj = Caser(args)
            obj.Args = args;
            obj.initialization();
        end

        function [ outputs ] = forward(obj, inputs)
            ac_conv = obj.Args.acconv;
            ac_fc = obj.Args.acfc;
            drop_rate = obj.Args.droprate;
            d = obj.Args.d;
            size_v = obj.Args.size_v; size_h = obj.Args.size_h;

            P = obj.Params.P; Q = obj.Params.Q;
            W1 = obj.Params.W1; W2 = obj.Params.W2;
            B1 = obj.Params.B1; B2 = obj.Params.B2;
            f_h = obj.Params.f_h; f_v = obj.Params.f_v;
            b_h = obj.Params.b_h; b_v = obj.Params.b_v;

            seq = inputs.seq; 
            targets = inputs.targets;
            negatives = inputs.negatives;
            user = inputs.user;

            pu = P(user, :);
            ql = Q(seq, :);

            % reshape to 4-D
            pu = reshape(pu, [1,1,d,1]);
            ql = reshape(ql, [length(seq),d,1,1]);

            % get i,j indice of W2,B2 to reduce computation
            w2i = W2(:,:,:,targets);
            w2j = W2(:,:,:,negatives);
            b2i = B2(targets);
            b2j = B2(negatives);

            flatten_layer = cell(2, 1);
            % apply filter: vertical
            res_v = cell(length(f_v), 2);
            for i=1:length(size_v)
                conv = vl_nnconv(ql, f_v{i}, b_v{i});
                % apply activation function
                conv_ac = conv;

                res_v{i,1} = conv;
                res_v{i,2} = conv_ac;
            end
            concat_v = vl_nnconcat(res_v(:,2), 1);
            flatten_v = reshape(concat_v, numel(concat_v), []);
            flatten_layer{1} = flatten_v;

            % apply filter: horizontal
            res_h = cell(length(f_h), 3);
            for i=1:length(size_h)
                conv = vl_nnconv(ql, f_h{i}, b_h{i});
                % apply activation function
                switch ac_conv
                    case 'iden'
                        conv_ac = conv;
                    case 'sigm'
                        conv_ac = vl_nnsigmoid(conv);
                    case 'tanh'
                        conv_ac = vl_nntanh(conv);
                    case 'relu'
                        conv_ac = vl_nnrelu(conv);
                end
                % max-pooling
                sizes = size(conv_ac);
                pool = vl_nnpool(conv_ac, [sizes(1), 1]);

                res_h{i,1} = conv;
                res_h{i,2} = conv_ac;
                res_h{i,3} = pool;
            end
            concat_h = vl_nnconcat(res_h(:,3), 3);
            flatten_h = reshape(concat_h, numel(concat_h), []);
            flatten_layer{2} = flatten_h;

            flatten = vl_nnconcat(flatten_layer, 1);
            % apply dropout
            if strcmp(mode, 'eval')
                drop_rate = 0;
            end
            [flatten_dropout, mask] = vl_nndropout(flatten, 'rate', drop_rate);
            c1 = vl_nnconv(flatten_dropout, W1, B1);
            % apply activation function
            switch ac_fc
                case 'iden'
                    z1 = c1;
                case 'sigm'
                    z1 = vl_nnsigmoid(c1);
                case 'tanh'
                    z1 = vl_nntanh(c1);
                case 'relu'
                    z1 = vl_nnrelu(c1);
            end

            z = vl_nnconcat({z1, pu}, 3);

            % output layer
            oi = vl_nnconv(z, w2i, b2i);
            oj = vl_nnconv(z, w2j, b2j);

            % outputs
            outputs.oi = oi;
            outputs.oj = oj;
            
            if strcmp(mode, 'train')
                % save intermediate variables for backward
                obj.Temp_Params.flatten_layer = flatten_layer;
                obj.Temp_Params.res_v = res_v;
                obj.Temp_Params.concat_v = concat_v;
                obj.Temp_Params.res_h = res_h;
                obj.Temp_Params.concat_h = concat_h;
                obj.Temp_Params.flatten = flatten;
                obj.Temp_Params.flatten_dropout = flatten_dropout;
                obj.Temp_Params.mask = mask;
                obj.Temp_Params.z1 = z1;
                obj.Temp_Params.c1 = c1;
                obj.Temp_Params.z = z;
            end
        end
        
        function backward(obj, inputs)
            doi = inputs.doi; doj = inputs.doj;
            seq = inputs.seq; targets = inputs.targets; negatives = inputs.negatives; user = inputs.user;
            
            d = obj.Args.d;
            size_v = obj.Args.size_v; size_h = obj.Args.size_h; 
            ac_conv = obj.Args.acconv; ac_fc = obj.Args.acfc;
            
            pu = obj.Params.P(user, :);
            ql = obj.Params.Q(seq, :);
            pu = reshape(pu, [1,1,d,1]);
            ql = reshape(ql, [length(seq),d,1,1]);

            w2i = obj.Params.W2(:,:,:,targets);
            w2j = obj.Params.W2(:,:,:,negatives);
            b2i = obj.Params.B2(targets);
            b2j = obj.Params.B2(negatives);

            W1 = obj.Params.W1;
            B1 = obj.Params.B1;
            f_v = obj.Params.f_v;
            f_h = obj.Params.f_h;
            b_v = obj.Params.b_v;
            b_h = obj.Params.b_h;
            
            % retrive saved intermediate variables for backward
            flatten_layer = obj.Temp_Params.flatten_layer;
            res_v = obj.Temp_Params.res_v;
            concat_v = obj.Temp_Params.concat_v;
            res_h = obj.Temp_Params.res_h;
            concat_h = obj.Temp_Params.concat_h;
            flatten = obj.Temp_Params.flatten;
            flatten_dropout = obj.Temp_Params.flatten_dropout;
            mask = obj.Temp_Params.mask;
            z1 = obj.Temp_Params.z1;
            c1 = obj.Temp_Params.c1;
            z = obj.Temp_Params.z;

            dql = zeros(size(ql));
            df_v = cell(length(size_v), 1);
            df_h = cell(length(size_h), 1);
            db_v = cell(length(size_v), 1);
            db_h = cell(length(size_h), 1);

            % output layer
            dz = 0;
            [dz_i, dw2i, db2i] = vl_nnconv(z, w2i, b2i, doi);
            dz = dz + dz_i;

            [dz_j, dw2j, db2j] = vl_nnconv(z, w2j, b2j, doj);
            dz = dz + dz_j;
            
            dzs = vl_nnconcat({z1,pu}, 3, dz);
            dz1 = dzs{1};
            dpu = dzs{2};

            % apply activation function
            switch ac_fc
                case 'iden'
                    dc1 = dz1;
                case 'sigm'
                    dc1 = vl_nnsigmoid(c1, dz1);
                case 'tanh'
                    dc1 = vl_nntanh(c1, dz1);
                case 'relu'
                    dc1 = vl_nnrelu(c1, dz1);
            end

            % apply dropout
            [dflatten_dropout, dW1, dB1] = vl_nnconv(flatten_dropout, W1, B1, dc1);
            dflatten = vl_nndropout(flatten, dflatten_dropout, 'mask', mask);
            dflatten_layer = vl_nnconcat(flatten_layer, 1, dflatten);

            % apply filter: horizontal
            dflatten_v = dflatten_layer{1};
            dconcat_v = reshape(dflatten_v, size(concat_v));
            dres_v = vl_nnconcat(res_v(:,2), 1, dconcat_v);

            for i=1:numel(size_v)
                dconv_ac = dres_v{i};
                [dql_v, df_v{i}, db_v{i}] = vl_nnconv(ql, f_v{i}, b_v{i}, dconv_ac);
                dql = dql + dql_v;
            end

            % apply filter: horizontal
            dflatten_h = dflatten_layer{2};
            dconcat_h = reshape(dflatten_h, size(concat_h));
            dres_h = vl_nnconcat(res_h(:,3), 3, dconcat_h);

            for i=1:numel(size_h)
                % max-pooling
                sizes = size(res_h{i,2});
                dpool = vl_nnpool(res_h{i,2}, [sizes(1), 1], dres_h{i});
                % apply activation function
                switch ac_conv
                    case 'iden'
                        dconv_ac = dpool;
                    case 'sigm'
                        dconv_ac = vl_nnsigmoid(res_h{i,1}, dpool);
                    case 'tanh'
                        dconv_ac = vl_nntanh(res_h{i,1}, dpool);
                    case 'relu'
                        dconv_ac = vl_nnrelu(res_h{i,1}, dpool);
                end
                [dql_h, df_h{i}, db_h{i}] = vl_nnconv(ql, f_h{i}, b_h{i}, dconv_ac);
                dql = dql + dql_h;
            end

            % accumulate gradients
            obj.Grads.P(user, :) = obj.Grads.P(user, :) + reshape(dpu, [1,length(dpu)]);
            obj.Grads.Q(seq, :) = obj.Grads.Q(seq, :) + dql;

            obj.Grads.W1 = obj.Grads.W1 + dW1;
            obj.Grads.B1 = obj.Grads.B1 + dB1;
            obj.Grads.W2(:,:,:,targets) = obj.Grads.W2(:,:,:,targets) + dw2i;
            obj.Grads.B2(targets) = obj.Grads.B2(targets) + db2i;
            obj.Grads.W2(:,:,:,negatives) = obj.Grads.W2(:,:,:,negatives) + dw2j;
            obj.Grads.B2(negatives) = obj.Grads.B2(negatives) + db2j;

            for i=1:length(size_v)
                obj.Grads.f_v{i} = obj.Grads.f_v{i} + df_v{i};
                obj.Grads.b_v{i} = obj.Grads.b_v{i} + db_v{i};
            end

            for i=1:length(size_h)
                obj.Grads.f_h{i} = obj.Grads.f_h{i} + df_h{i};
                obj.Grads.b_h{i} = obj.Grads.b_h{i} + db_h{i};
            end
        end

        function updates(obj, config)
            config.learning_rate = obj.Args.lrate; config.l2 = obj.Args.l2;
            [obj.Params.P, obj.States.P] = adam_update(obj.Params.P, obj.Grads.P, config, obj.States.P);
            [obj.Params.Q, obj.States.Q] = adam_update(obj.Params.Q, obj.Grads.Q, config, obj.States.Q);

            [obj.Params.W1, obj.States.W1] = adam_update(obj.Params.W1, obj.Grads.W1, config, obj.States.W1);
            [obj.Params.B1, obj.States.B1] = adam_update(obj.Params.B1, obj.Grads.B1, config, obj.States.B1);
            [obj.Params.W2, obj.States.W2] = adam_update(obj.Params.W2, obj.Grads.W2, config, obj.States.W2);
            [obj.Params.B2, obj.States.B2] = adam_update(obj.Params.B2, obj.Grads.B2, config, obj.States.B2);

            for i=1:length(obj.Params.f_v)
                [obj.Params.f_v{i}, obj.States.f_v{i}] = adam_update(obj.Params.f_v{i}, obj.Grads.f_v{i}, config, obj.States.f_v{i});
                [obj.Params.b_v{i}, obj.States.b_v{i}] = adam_update(obj.Params.b_v{i}, obj.Grads.b_v{i}, config, obj.States.b_v{i});
            end

            for i=1:length(obj.Params.f_h)
                [obj.Params.f_h{i}, obj.States.f_h{i}] = adam_update(obj.Params.f_h{i}, obj.Grads.f_h{i}, config, obj.States.f_h{i});
                [obj.Params.b_h{i}, obj.States.b_h{i}] = adam_update(obj.Params.b_h{i}, obj.Grads.b_h{i}, config, obj.States.b_h{i});
            end
            % the first row of Q is for Padding
            obj.Params.Q(1,:) = 0;
        end
        
        function zero_grads(obj)
            size_v = obj.Args.size_v; size_h = obj.Args.size_h;

            obj.Grads.P = zeros(size(obj.Grads.P));
            obj.Grads.Q = zeros(size(obj.Grads.Q));

            obj.Grads.W1 = zeros(size(obj.Grads.W1));
            obj.Grads.W2 = zeros(size(obj.Grads.W2));
            obj.Grads.B1 = zeros(size(obj.Grads.B1));
            obj.Grads.B2 = zeros(size(obj.Grads.B2));

            for i=1:length(size_v)
                obj.Grads.f_v{i} = zeros(size(obj.Grads.f_v{i}));
                obj.Grads.b_v{i} = zeros(size(obj.Grads.b_v{i}));
            end
            for i=1:length(size_h)
                obj.Grads.f_h{i} = zeros(size(obj.Grads.f_h{i}));
                obj.Grads.b_h{i} = zeros(size(obj.Grads.b_h{i}));
            end
        end
        
        function normalize_grads(obj, down_factor)
            size_v = obj.Args.size_v; size_h = obj.Args.size_h;
            
            obj.Grads.P = obj.Grads.P / down_factor;
            obj.Grads.Q = obj.Grads.Q / down_factor;

            obj.Grads.W1 = obj.Grads.W1 / down_factor;
            obj.Grads.W2 = obj.Grads.W2 / down_factor;
            obj.Grads.B1 = obj.Grads.B1 / down_factor;
            obj.Grads.B2 = obj.Grads.B2 / down_factor;

            for i=1:length(size_h)
                obj.Grads.f_h{i} = obj.Grads.f_h{i} / down_factor;
                obj.Grads.b_h{i} = obj.Grads.b_h{i} / down_factor;
            end

            for i=1:length(size_v)
                obj.Grads.f_v{i} = obj.Grads.f_v{i} / down_factor;
                obj.Grads.b_v{i} = obj.Grads.b_v{i} / down_factor;
            end
        end

        function initialization(obj)
            n_users = obj.Args.n_users;
            n_items = obj.Args.n_items;
            d = obj.Args.d;
            nv = obj.Args.nv;
            nh = obj.Args.nh;
            L = obj.Args.L;
            
            size_v = 1;
            size_h = 1:L;
            obj.Args.size_v = size_v;
            obj.Args.size_h = size_h;
            
            % user and item latent factors
            P = normrnd(0, 1/d, [n_users, d]);
            Q = normrnd(0, 1/d, [n_items + 1, d]);  % add one more row for padding
            Q(1,:) = 0;

            % CNN filters
            f_v = cell(length(size_v), 1);
            f_h = cell(length(size_h), 1);
            b_v = cell(length(size_v), 1);
            b_h = cell(length(size_h), 1);

            temp = 0;
            % vertical filter
            temp = temp + d*nv;
            for i=1:length(size_v)
                norm = 1 / sqrt(L*size_v(i));
                f_v{i} = unifrnd(-norm, norm, [L, size_v(i), 1, nv]);
                b_v{i} = unifrnd(-norm, norm, [1, nv]);
            end

            % horizontal filter
            temp = temp + length(size_h) * nh;  % due to the max-pooling
            for i=1:length(size_h)
                norm = 1 / sqrt(d*size_h(i));
                f_h{i} = unifrnd(-norm, norm, [size_h(i), d, 1, nh]);
                b_h{i} = unifrnd(-norm, norm, [1, nh]);
            end

            % fc layers
            W1_in = temp;
            W1_out = d;
            W2_in = W1_out+d;
            W2_out = n_items;

            % W1,W2 and B1,B2
            norm = 1 / sqrt(W1_in);
            W1 = unifrnd(-norm, norm, [W1_in, 1, 1, W1_out]); % use 4-D tensor for matconvnet
            B1 = unifrnd(-norm, norm, [W1_out, 1]);
            W2 = normrnd(0, 1/W2_in, [1, 1, W2_in, W2_out]); % use 4-D tensor for matconvnet 
            B2 = zeros([W2_out, 1]);

            params.P = P; params.Q = Q;
            params.W1 = W1; params.W2 = W2;
            params.B1 = B1; params.B2 = B2;
            params.f_h = f_h; params.f_v = f_v;
            params.b_h = b_h; params.b_v = b_v;

            % init states for ADAM
            states.P = struct();
            states.Q = struct();

            states.W1 = struct();
            states.B1 = struct();
            states.W2 = struct();
            states.B2 = struct();

            for i=1:length(size_v)
                states.f_v{i} = struct();
                states.b_v{i} = struct();
            end
            for i=1:length(size_h)
                states.f_h{i} = struct();
                states.b_h{i} = struct();
            end

            % init grads for ADAM
            grads.P = zeros(size(P));
            grads.Q = zeros(size(Q));

            grads.W1 = zeros(size(W1));
            grads.W2 = zeros(size(W2));
            grads.B1 = zeros(size(B1));
            grads.B2 = zeros(size(B2));

            grads.f_v = cell(length(f_v), 1);
            grads.f_h = cell(length(f_h), 1);
            grads.b_v = cell(length(f_v), 1);
            grads.b_h = cell(length(f_h), 1);

            for i=1:length(size_v)
                grads.f_v{i} = zeros(size(f_v{i}));
                grads.b_v{i} = zeros(size(b_v{i}));
            end
            for i=1:length(size_h)
                grads.f_h{i} = zeros(size(f_h{i}));
                grads.b_h{i} = zeros(size(b_h{i}));
            end
            
            obj.Params = params;
            obj.Grads = grads;
            obj.States = states;
        end
        
    end
    
end


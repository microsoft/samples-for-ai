import tensorflow as tf
import numpy as np
import mycommon as mc

class BAGRNN_Model:
    def __init__(self,
                 bag_num = 50,
                 enc_dim = 256,
                 embed_dim = 200,
                 rel_dim = None,
                 cat_n = 5,
                 sent_len = 120,
                 word_n = 80000,
                 extra_n = 3,
                 word_embed = None,
                 dropout = None,
                 cell_type = 'gru',
                 adv_eps = None,
                 adv_type = 'sent',
                 tune_embed = False,
                 use_softmax_loss = None,
                 sampled_sigmoid_loss = False,
                 max_dist_embed = None,
                 excl_na_loss = True,
                 only_perturb_pos_rel = False):
        self.bag_num = bag_num # total number of bags
        self.enc_dim = enc_dim
        if rel_dim is None:
            self.rel_dim = 3 * enc_dim if cell_type == 'pcnn' else 2 * enc_dim
        else:
            self.rel_dim = rel_dim
        self.embed_dim = embed_dim
        self.cat_n = cat_n
        self.sent_len = sent_len
        self.pretrain_word_embed = word_embed
        self.word_n = word_n
        self.extra_n = extra_n
        self.dropout = dropout
        self.cell_type = cell_type
        self.adv_eps = adv_eps  # eps for adversarial training, if None, classical feedfwd net
        self.adv_type = adv_type  # type of adversarial perturbation: batch, bag, sent
        self.tune_embed = tune_embed
        self.use_softmax_loss = (use_softmax_loss is not None)  # whether to use softmax loss or sigmoid loss
        self.use_full_softmax = self.use_softmax_loss and (use_softmax_loss > 0)
        self.sampled_sigmoid_loss = sampled_sigmoid_loss and not use_softmax_loss
        self.max_dist_embed = max_dist_embed
        self.use_pcnn = (cell_type == 'pcnn')  # None for RNN; other wise use PCNN with feature size <use_pcnn>
        self.excl_na_loss = excl_na_loss  # exclude NA in the loss function, only effective for sigmoid loss
        self.only_perturb_pos_rel = only_perturb_pos_rel

    def build(self, is_training,
              ent_dim = 3,
              dropout_embed = True):
        self.is_training = is_training

        bag_num = self.bag_num
        cat_n = self.cat_n
        L = self.sent_len
        rel_dim = self.rel_dim
        enc_dim = self.enc_dim
        cell_type = self.cell_type
        dropout = self.dropout

        ###################################
        # create placeholders
        #####
        # data shape info
        self.shapes = shapes = tf.placeholder(tf.int32, [self.bag_num + 1])
        # input data
        self.X = tf.placeholder(tf.int32, [None, L])
        self.ent = tf.placeholder(tf.int32, [None, L])
        if self.max_dist_embed is not None:
            self.ent2 = tf.placeholder(tf.int32, [None, L])
        # labels
        self.Y = ph_Y = tf.placeholder(tf.float32, [bag_num, cat_n])
        # sentence length
        self.length = length = tf.placeholder(tf.int32, [None])
        # sentence mask
        self.mask = mask = tf.placeholder(tf.float32, [None, L])
        # adversarial eps
        if self.adv_eps is not None:
            self.adv_eps = tf.placeholder(tf.float32, shape=())
        # loss mask
        if self.sampled_sigmoid_loss:
            self.loss_mask = loss_mask = tf.placeholder(tf.float32, [bag_num, cat_n])
        else:
            loss_mask = None
        if self.use_pcnn:
            self.pcnn_mask = tf.placeholder(tf.float32, [None, 3, L])
            pcnn_pos_mask = tf.expand_dims(tf.transpose(self.pcnn_mask, [0, 2, 1]), axis=1)  # [batch, 1, L, 3]
        if self.use_full_softmax:
            self.diag = tf.expand_dims(tf.eye(cat_n, dtype=tf.float32), axis=0)

        #################################
        # create embedding variables
        ####
        self.exclude_clip_vars = set()
        # pre-process entity embedding
        if self.max_dist_embed is None:
            self.ent_embed = tf.constant(np.array([[0] * ent_dim * 2, [1] * ent_dim + [0] * ent_dim, [0] * ent_dim + [1] * ent_dim],
                                                  dtype=np.float32),
                                         dtype=tf.float32)
        else:
            self.ent_embed = tf.get_variable('dist_embed', [2 * self.max_dist_embed + 1, ent_dim],
                                             initializer=tf.random_normal_initializer(0, 0.01))
            self.exclude_clip_vars.add(self.ent_embed)
        if self.pretrain_word_embed is not None:
            if self.tune_embed:
                pretrain_embed = tf.get_variable('pretrain_embed',
                                                 initializer=self.pretrain_word_embed)
                self.exclude_clip_vars.add(pretrain_embed)
            else:
                pretrain_embed = tf.constant(self.pretrain_word_embed,dtype=tf.float32)
            extra_embed = tf.get_variable('extra_embed', [self.extra_n,
                                                          self.embed_dim],
                                          initializer=tf.random_normal_initializer(0,0.01))
            self.exclude_clip_vars.add(extra_embed)
            self.word_embed = tf.concat([pretrain_embed, extra_embed], axis=0)
        else:
            self.word_embed = tf.get_variable('word_embed', [self.word_n+self.extra_n,
                                                             self.embed_dim],
                                              initializer=tf.random_normal_initializer(0,0.01))
            self.exclude_clip_vars.add(self.word_embed)

        ################################
        # discriminative model
        #####
        self.orig_inputs = orig_inputs = mc.get_embedding(self.X, self.word_embed,
                                                          self.dropout if dropout_embed else None, self.is_training)
        if self.max_dist_embed is not None:
            dist1_embed = mc.get_embedding(self.ent, self.ent_embed,
                                           self.dropout if dropout_embed else None, self.is_training)
            dist2_embed = mc.get_embedding(self.ent2, self.ent_embed,
                                           self.dropout if dropout_embed else None, self.is_training)
            ent_inputs = tf.concat([dist1_embed, dist2_embed], axis=2)
        else:
            ent_inputs = mc.get_embedding(self.ent, self.ent_embed)  # [batch,L,dim]

        use_softmax_loss = self.use_softmax_loss
        use_full_softmax = self.use_full_softmax
        use_pcnn = self.use_pcnn
        pcnn_feat_size = self.enc_dim

        def discriminative_net(word_inputs, name = 'discriminative-net', reuse = False,
                                 only_pos_rel_loss = False):
            with tf.variable_scope(name, reuse=reuse):
                if only_pos_rel_loss:
                    pos_rel_mask = ph_Y
                    # when y = [0, 0, ..., 0]: pos_rel_mask = [1, 1, ..., 1]
                    # o.w. pos_rel_mask = y
                    #na_flag = 1 - tf.reduce_max(ph_Y, axis=1, keep_dims=True)
                    #pos_rel_mask = ph_Y + na_flag

                inputs = tf.concat([word_inputs, ent_inputs], axis = 2)  # [batch, L, dim]

                if not use_pcnn:  # use RNN
                    outputs, states = mc.mybidrnn(inputs, length, enc_dim,
                                                  cell_name = cell_type,
                                                  scope = 'bidirect-rnn')
                    # sentence information
                    V = tf.concat(states, axis=1) # [batch, rel_dim]
                    V_dim = enc_dim * 2
                else:
                    # use pcnn
                    feat_size = pcnn_feat_size
                    window_size = 3
                    inputs = tf.expand_dims(inputs, axis=1)  # [batch, 1, L, dim]
                    conv_out = tf.squeeze(tf.nn.relu(
                        tf.layers.conv2d(inputs, feat_size, [1, window_size], 1, padding='same',
                                         kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
                    ))  # [batch, L, feat_size]
                    conv_out = tf.expand_dims(tf.transpose(conv_out, [0, 2, 1]), axis=-1)  # [batch, feat, L, 1]
                    pcnn_pool = tf.reduce_max(conv_out * pcnn_pos_mask, axis=2)  # [batch, feat, 3]
                    V = tf.reshape(pcnn_pool, [-1, feat_size * 3])
                    V_dim = feat_size * 3

                if V_dim != rel_dim:
                    V = mc.linear(V, rel_dim, scope='embed_proj')
                if dropout:
                    V = tf.layers.dropout(V, rate=dropout,
                                          training=is_training)

                #################################
                # Multi Label Multi Instance Learning
                #####
                Q = tf.get_variable('relation_embed', [rel_dim, cat_n],
                                    initializer=tf.random_normal_initializer(0, 0.01))
                if use_full_softmax:
                    A = tf.get_variable('classify-proj', [rel_dim, cat_n],
                                        initializer=tf.random_normal_initializer(0, 0.01))
                else:
                    A = tf.get_variable('classify-proj', [cat_n, rel_dim],
                                        initializer=tf.random_normal_initializer(0, 0.01))
                alpha = tf.matmul(tf.nn.tanh(V), Q) # [batch, cat_n]
                # process bags
                logits_list = []
                for i in range(bag_num):
                    n = shapes[i+1] - shapes[i]
                    curr_V = V[shapes[i]:shapes[i+1], :]  # [n, rel_dim]
                    curr_alpha = alpha[shapes[i]:shapes[i+1], :]  # [n, cat_n]
                    weight = tf.nn.softmax(tf.transpose(curr_alpha, [1, 0]))  # [cat_n, n]
                    full_weight = tf.tile(tf.expand_dims(weight, axis=-1), [1, 1, rel_dim])  # [cat_n, n, dim]
                    full_V = tf.tile(tf.expand_dims(curr_V, axis=0), [cat_n, 1, 1])
                    V_att = tf.reduce_sum(full_weight * full_V, axis=1)  # [cat_n, dim]
                    if use_full_softmax:
                        cat_logits = tf.matmul(V_att, A)  # [cat_n, cat_n]
                    else:
                        cat_logits = tf.reduce_sum(V_att * A, axis=1)  # [cat_n]
                    logits_list.append(cat_logits)
                logits = tf.stack(logits_list)  # [bag_num, cat_n] or [bag_num, cat_n, cat_n]
                if use_softmax_loss:
                    probs = tf.nn.softmax(logits)
                    if use_full_softmax:
                        # probs: [bag_num, cat_n, cat_n] last dimension normalized
                        probs = tf.reduce_sum(probs * self.diag, axis=-1)  # [bag_num, cat_n]
                        # optimize the sum of softmax-loss for each positive rel
                        loss = -tf.reduce_mean(tf.reduce_sum(tf.log(probs + 1e-20) * ph_Y, axis=1))
                    else:
                        # add all the probs, output the joint log probability
                        loss = -tf.reduce_mean(tf.log(tf.reduce_sum(probs * ph_Y, axis=1) + 1e-20))
                else:
                    probs = tf.nn.sigmoid(logits)
                    if loss_mask is not None:
                        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_Y, logits=logits)
                        if only_pos_rel_loss:
                            loss = loss * pos_rel_mask
                        loss = tf.reduce_sum(loss * loss_mask, axis=1)
                        weight = tf.reduce_sum(loss_mask, axis=1)
                        # weighted average of the individual sigmoid loss, rescale to full_sigmoid_loss
                        #coef = cat_n - 1 if self.excl_na_loss else cat_n
                        #loss = tf.reduce_mean(loss / weight) * coef
                        loss = tf.reduce_mean(loss) # * coef
                    else:
                        if self.excl_na_loss:
                            loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=ph_Y, logits=logits)  # [bag, cat_n]
                            if only_pos_rel_loss:
                                loss = loss * pos_rel_mask
                            loss = loss[:, 1:]  # exclude NA
                            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))
                        else:
                            if only_pos_rel_loss:
                                loss = tf.losses.sigmoid_cross_entropy(ph_Y, logits, weights=pos_rel_mask)
                            else:
                                loss = tf.losses.sigmoid_cross_entropy(ph_Y, logits)
            return probs, loss

        self.probs, self.raw_loss = discriminative_net(orig_inputs, reuse=False,
                                                       only_pos_rel_loss=(self.adv_eps is not None) and (self.only_perturb_pos_rel) and (not use_softmax_loss))
        if self.adv_eps is None:
            self.loss = self.raw_loss
        else:  # adversarial training
            raw_perturb = tf.gradients(self.raw_loss, orig_inputs)[0]  # [batch, L, dim]
            if self.adv_type == 'sent':
                # normalize per sentence
                self.perturb = perturb = self.adv_eps * tf.stop_gradient(
                    tf.nn.l2_normalize(raw_perturb * tf.expand_dims(mask, axis=-1), dim=[1, 2]))
            elif self.adv_type == 'batch':
                # normalize the whole batch
                self.perturb = perturb = self.adv_eps * tf.stop_gradient(
                    tf.nn.l2_normalize(raw_perturb * tf.expand_dims(mask, axis=-1), dim=[0,1,2]))
            else:  # bag-level normalization
                raw_perturb = tf.stop_gradient(raw_perturb * tf.expand_dims(mask, axis=-1))  # [batch, L, dim]
                perturb_list = []
                for i in range(bag_num):
                    curr_pt = raw_perturb[shapes[i]:shapes[i+1], :, :]  # [bag_size, L, dim]
                    perturb_list.append(self.adv_eps * tf.nn.l2_normalize(curr_pt, dim=[0,1,2]))
                self.perturb = perturb = tf.concat(perturb_list, axis=0)  # [batch, L, dim]
            self.perturb_inputs = perturb_inputs = orig_inputs + perturb
            self.perturb_probs, self.loss = discriminative_net(perturb_inputs, reuse=True)  # optimize the loss with perturbed loss

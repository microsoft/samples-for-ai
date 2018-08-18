import tensorflow as tf
import numpy as np
import time
import pickle
import mycommon as mc


class MyLogger:
    def __init__(self, logdir, clear_file = False):
        self.fname = logdir + '/progress.txt'
        if clear_file:
            mc.remove_file(self.fname)

    def print_model_spec(self, model):
        with open(self.fname, 'a') as f:
            print('+++++++++++++ Model Specs +++++++++++', file=f)
            if model.use_full_softmax:
                s = 'full-softmax-loss with positive rels'
            elif model.use_softmax_loss:
                s = 'combined-single-softmax-loss'
            elif model.sampled_sigmoid_loss is not None:
                if model.sampled_sigmoid_loss > 0:
                    s = 'sampled-sigmoid-loss <K={}>'.format(model.sampled_sigmoid_loss)
                else:
                    s = 'weighted-sigmoid-loss <C={}>'.format(-model.sampled_sigmoid_loss)
            else:
                s = 'full-sigmoid-loss'
            print('Loss: {}'.format(s), file=f)
            print('Model-Type: {}'.format(model.cell_type), file=f)
            print('Kernel-Size: {}'.format(model.enc_dim), file=f)
            print('Feat-Size: {}'.format(model.rel_dim), file=f)
            if model.max_dist_embed is not None:
                s = 'relative-pos-embed <max_dis={}>'.format(model.max_dist_embed)
            else:
                s = 'one-hot-pos'
            print('Entity-Pos-Embed: {}'.format(s), file=f)
            print('+++++++++++++++++++++++++++++++++++++', file=f)

    def print(self, str, to_screen = True):
        if to_screen:
            print(str)
        with open(self.fname, 'a') as f:
            print(str, file=f)


class BagTrainer:
    def __init__(self, model, loader,
                 lrate = 0.001,
                 clip_grad = None,
                 lrate_decay_step = 0,
                 sampled_loss = None,
                 adv_eps = None):
        self.model = model
        self.loader = loader
        self.lrate = lrate
        self.lrate_decay = lrate_decay_step
        self.clip_grad = clip_grad
        self.max_len = self.model.sent_len
        self.sampled_loss = sampled_loss
        self.adv_eps = adv_eps
        self.excl_na_loss = model.excl_na_loss
        tf.reset_default_graph()
        self.is_training = tf.placeholder(tf.bool)

    def update_learning_param(self):
        self.lrate = self.lrate * 0.9998

    def compute_relative_dist(self, batch_size, L, length, pos, max_dist):
        def calc_dist(x, a, b, cap):
            dis = 0
            if x <= a:
                dis = x-a
            elif x >= b:
                dis = x-b+1
            if dis < -cap:
                dis = -cap
            if dis > cap:
                dis = cap
            return dis + cap
        D1 = np.zeros((batch_size, L), dtype=np.int32)
        D2 = np.zeros((batch_size, L), dtype=np.int32)
        for i in range(batch_size):
            cur_len = int(length[i])
            for j in range(cur_len):
                D1[i, j] = calc_dist(j, pos[i][0], pos[i][1], max_dist)
                D2[i, j] = calc_dist(j, pos[i][2], pos[i][3], max_dist)
        return D1, D2

    def compute_pcnn_pool_mask(self, batch_size, L, length, pos):
        mask = np.zeros((batch_size, 3, L), dtype=np.float32)
        for i in range(batch_size):
            a,b,c,d = pos[i]
            if d <= a:
                c,d,a,b = pos[i]  # ensure ~ a<b<=c<d
            # piecewise cnn: 0...b-1; b ... d-1; d ... L
            if b>0:
                mask[i, 0, :b] = 1
            if b < d:
                mask[i, 1, b:d] = 1
            if d < length[i]:
                mask[i, 2, d:length[i]] = 1
        return mask

    def feed_dict(self, source='train'):
        loader = self.loader
        fd = dict(zip([self.is_training, self.learning_rate],
                      [(source == 'train'), self.lrate]))
        self.effective, X, Y, E, length, shapes, mask = loader.next_batch(source)

        self.batch_size = X.shape[0]

        M = self.model
        # feed size
        fd[M.shapes] = shapes
        fd[M.X] = X
        fd[M.Y] = Y
        fd[M.length] = length
        fd[M.mask] = mask

        if self.adv_eps is not None:
            fd[M.adv_eps] = self.adv_eps

        if M.max_dist_embed is None:
            fd[M.ent] = E
        else:
            D1, D2 = self.compute_relative_dist(self.batch_size, self.max_len,
                                                length, loader.cached_pos, M.max_dist_embed)
            fd[M.ent] = D1
            fd[M.ent2] = D2

        if M.use_pcnn:
            pcnn_mask = self.compute_pcnn_pool_mask(self.batch_size, self.max_len,
                                                    length, loader.cached_pos)
            fd[M.pcnn_mask] = pcnn_mask

        if self.sampled_loss is not None:
            bag_size = Y.shape[0]
            cat_n = Y.shape[1]
            loss_mask = Y.copy()
            if self.excl_na_loss:
                loss_mask[:, 0] = 0  # exclude NA loss
            if self.sampled_loss > 0:  # sample
                for i in range(bag_size):
                    idx = []
                    c = self.sampled_loss
                    for j in range(cat_n):
                        if self.excl_na_loss and j == 0:
                            continue  # ignore NA
                        if Y[i,j] < 0.5:
                            idx.append(j)  # negative rel
                        else:
                            c -= 1  # positive rel
                    np.random.shuffle(idx)
                    for j in range(c):
                        loss_mask[i, idx[j]] = 1
            else:  # normalize weights for zero labels
                scale = -self.sampled_loss
                for i in range(bag_size):
                    idx = []
                    pos = 0
                    neg = 0
                    for j in range(cat_n):
                        if self.excl_na_loss and j == 0:
                            continue
                        if Y[i,j] < 0.5:
                            idx.append(j)
                            neg += 1
                        else:
                            pos += 1
                    #weight = 1 / neg * scale
                    if pos == 0:
                        weight = 1  # NA relation
                    else:
                        weight = pos / neg * scale
                    for j in idx:
                        loss_mask[i, j] = weight
            fd[M.loss_mask] = loss_mask
        return fd

    def evaluate(self, sess, stats_file = './stats/eval_stats.pkl', max_relations = None, incl_conf = False):
        logger = self.logger
        logger.print('Evaluating ...')
        ts = time.time()
        loader = self.loader
        M = self.model
        n_bag = loader.get_bag_n()
        n_rel = len(loader.rel_name) # assume NA is 0
        rel_conf = []
        pos_rel_n = 0
        k = 0
        while k < n_bag:
            all_conf = sess.run(M.probs, feed_dict=self.feed_dict('eval'))
            m = self.effective
            assert(m>0)
            for i in range(m):
                conf = all_conf[i, :]
                info = loader.get_bag_info(k + i)
                for j in range(1, n_rel):
                    flag = 1 if j in info else 0
                    pos_rel_n += flag
                    rel_conf.append((j, conf[j], flag))
            k += m
        rel_conf.sort(key=lambda x: x[1], reverse=True)
        precis = []
        recall = []
        f1_score = []
        tar_conf = []
        correct = 0

        def get_f1(p, r):
            if p + r <= 1e-20:
                return 0
            return 2 * p * r / (p + r)

        #if max_relations is not None:
        #    rel_conf = rel_conf[:max_relations]

        for i, dat in enumerate(rel_conf):
            r, p, f = dat
            correct += f
            if f > 0:
                precis.append(correct / (i + 1))
                recall.append(correct / pos_rel_n)
                f1_score.append(get_f1(precis[-1], recall[-1]))
                tar_conf.append(p)
            if (max_relations is not None) and i+1 == max_relations:
                m = len(precis)

        auc = np.mean(precis)
        if max_relations is not None:
            precis = precis[:m]
            recall = recall[:m]
            f1_score = f1_score[:m]
            tar_conf = tar_conf[:m]

        data_to_dump = [precis, recall, f1_score]
        if incl_conf:
            data_to_dump.append(tar_conf)
        with open(stats_file, 'wb') as f:
            pickle.dump(data_to_dump, f)
        best_f1 = max(f1_score)
        logger.print('  -> Done! Time Elapsed = {}s'.format(time.time()-ts))
        logger.print('>>>> Best F1 Score = %.5f' % best_f1)
        logger.print('>>>> AUC (full) = %.5f' % auc)

    def train(self,
              name = 'bagrnn',
              epochs = 10,
              log_dir = './log',
              model_dir = './model',
              stats_dir = './stats',
              restore_dir = None,
              test_gap = 5,
              report_rate = 0.5,
              gpu_usage = 0.9,
              max_eval_rel = None):
        # [NOTE] assume model is already built
        loader = self.loader
        self.logger = logger = MyLogger(stats_dir)
        logger.print("\n\n\n\n", False)
        logger.print(">>>>>>>>>>>>>>>>> New Start <<<<<<<<<<<<<<<<<<<<<", False)
        logger.print_model_spec(self.model)

        # build tensorboard monitoring variables
        tf.summary.scalar('miml-loss', self.model.raw_loss)
        if self.model.adv_eps is not None:
            tf.summary.scalar('adversarial-loss', self.model.loss)
        self.merged = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(log_dir + '/train')
        self.test_writer = tf.summary.FileWriter(log_dir + '/test')

        # training related
        self.loss = self.model.loss
        self.learning_rate = tf.placeholder(tf.float32)
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.clip_grad is None:
            self.train_op = optimizer.minimize(self.loss)
        else:
            self.train_op = mc.minimize_and_clip(optimizer, self.loss, clip_val=self.clip_grad,
                                                 exclude=self.model.exclude_clip_vars)

        # Training
        total_batches = loader.train_batches  # total batches
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_usage)
        config = tf.ConfigProto(allow_soft_placement=True,
                                gpu_options=gpu_options)
        saver = tf.train.Saver(max_to_keep=epochs//2+1)  # keep the last half of the models
        global_batch_counter = 0
        global_ep_counter = 0

        with tf.Session(config=config) as sess:
            try:
                if restore_dir is None:
                    sess.run(tf.global_variables_initializer())
                else:
                    logger.print('Warmstart from {} ...'.format(restore_dir))
                    saver.restore(sess, restore_dir)
                    with open(restore_dir+'-counter.pkl','rb') as f:
                        global_batch_counter, global_ep_counter = pickle.load(f)

                # Run Training

                accu_batch = 0
                accu_loss = 0

                if epochs <= 0:
                    # only perform evaluation
                    loader.new_epoch()
                    self.evaluate(sess, stats_dir + '/{m}_eval_stats_full.pkl'.format(m=name), incl_conf=True)

                for ep in range(epochs):
                    global_ep_counter += 1
                    iter_n = 0
                    ts = time.time()
                    cur_rate = report_rate
                    logger.print ('>>> Current Starting Epoch#{}'.format(global_ep_counter))
                    loader.new_epoch()
                    # FOR DEBUG
                    #if ep == 0:
                    #    self.evaluate(sess, stats_dir + '/random_eval_stats.pkl')
                    #    exit(0)
                    for i in range(total_batches):
                        iter_n += 1
                        global_batch_counter += 1
                        summary, c_loss, _ = \
                            sess.run([self.merged, self.model.loss, self.train_op],
                                     feed_dict=self.feed_dict('train'))
                        accu_batch += 1
                        accu_loss += c_loss
                        self.train_writer.add_summary(summary, global_batch_counter)
                        if iter_n % test_gap == 0:
                            summary = sess.run(self.merged,
                                               feed_dict=self.feed_dict('test'))
                            self.test_writer.add_summary(summary, global_batch_counter)

                        if iter_n >= total_batches * cur_rate:
                            logger.print(' --> {x} / {y} finished! ratio = {r},   elapsed = {t}'.format(
                                x = iter_n,
                                y = total_batches,
                                r = (1.0 * iter_n) / total_batches,
                                t = time.time() - ts))
                            logger.print('   > loss = %.6f' % (accu_loss / accu_batch))
                            accu_batch = 0
                            accu_loss = 0
                            cur_rate += report_rate

                        if self.lrate_decay > 0 and \
                           global_batch_counter % self.lrate_decay == 0:
                            self.update_learning_param()  # learning rate decay

                    save_path = saver.save(sess,
                                           model_dir+'/{m}_ep{i}'.format(m=name,i=global_ep_counter))
                    with open(model_dir+'/{m}_ep{i}-counter.pkl'.format(m=name,i=global_ep_counter), 'wb') as f:
                        pickle.dump([global_batch_counter, global_ep_counter], f)
                    logger.print("Model saved in file: %s" % save_path)
                    self.evaluate(sess, stats_dir + '/{m}_eval_stats_ep{i}.pkl'.format(m=name, i=global_ep_counter),
                                  max_relations=max_eval_rel)
            except KeyboardInterrupt:
                logger.print('Training Interrupt!!')
                save_path = saver.save(sess,
                                       model_dir+'/{m}_killed_iter{i}'.format(m=name,i=global_batch_counter))
                with open(model_dir+'/{m}_killed_iter{i}-counter.pkl'.format(m=name,i=global_batch_counter), 'wb') as f:
                    pickle.dump([global_batch_counter, global_ep_counter], f)
                logger.print('Model saved in file: %s' % save_path)

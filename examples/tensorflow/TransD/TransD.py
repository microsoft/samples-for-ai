#coding:utf-8

import os
import time
import datetime
import ctypes

import numpy as np
import tensorflow as tf

lib = ctypes.cdll.LoadLibrary("./init_dll.dll")
test_lib = ctypes.cdll.LoadLibrary("./test_dll.dll")

def init_config(raw_path=b"./data/FB15K/", is_test=False, load_curr_model=False):
	cstr_path = ctypes.c_char_p(raw_path)
	lib.set_in_path(cstr_path, len(raw_path))
	test_lib.set_in_path(cstr_path, len(raw_path))
	return {
		'test_flag' : is_test,
		'load_from_data' : load_curr_model,
		'L1_flag' : True,
		'hidden_size' : 50,
		'nbatches' : 100,
		'entity' : 0,
		'relation' : 0,
		'train_times' : 500,
		'margin' : 1.0
	}


def model_calc(e, t, r):
    return e + tf.reduce_sum(e*t, 1, keep_dims=True)*r

def init_model(config_dict):
    pos_h = tf.placeholder(tf.int32, [None])
    pos_t = tf.placeholder(tf.int32, [None])
    pos_r = tf.placeholder(tf.int32, [None])

    neg_h = tf.placeholder(tf.int32, [None])
    neg_t = tf.placeholder(tf.int32, [None])
    neg_r = tf.placeholder(tf.int32, [None])

    entity_total = config_dict['entity']
    relation_total = config_dict['relation']
    batch_size = config_dict['batch_size']
    size = config_dict['hidden_size']
    margin = config_dict['margin']

    with tf.name_scope("embedding"):
        ent_embeddings = tf.get_variable(
            name = "ent_embedding", shape = [entity_total, size], 
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            )
        rel_embeddings = tf.get_variable(
            name = "rel_embedding", shape = [relation_total, size], 
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            )
        ent_transfer = tf.get_variable(
            name = "ent_transfer", shape = [entity_total, size], 
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            )
        rel_transfer = tf.get_variable(
            name = "rel_transfer", shape = [relation_total, size], 
            initializer = tf.contrib.layers.xavier_initializer(uniform = False)
            )
        
        pos_h_e = tf.nn.embedding_lookup(ent_embeddings, pos_h)
        pos_t_e = tf.nn.embedding_lookup(ent_embeddings, pos_t)
        pos_r_e = tf.nn.embedding_lookup(rel_embeddings, pos_r)
        pos_h_t = tf.nn.embedding_lookup(ent_transfer, pos_h)
        pos_t_t = tf.nn.embedding_lookup(ent_transfer, pos_t)
        pos_r_t = tf.nn.embedding_lookup(rel_transfer, pos_r)
        neg_h_e = tf.nn.embedding_lookup(ent_embeddings, neg_h)
        neg_t_e = tf.nn.embedding_lookup(ent_embeddings, neg_t)
        neg_r_e = tf.nn.embedding_lookup(rel_embeddings, neg_r)
        neg_h_t = tf.nn.embedding_lookup(ent_transfer, neg_h)
        neg_t_t = tf.nn.embedding_lookup(ent_transfer, neg_t)
        neg_r_t = tf.nn.embedding_lookup(rel_transfer, neg_r)

        pos_h_e = model_calc(pos_h_e, pos_h_t, pos_r_t)
        pos_t_e = model_calc(pos_t_e, pos_t_t, pos_r_t)
        neg_h_e = model_calc(neg_h_e, neg_h_t, neg_r_t)
        neg_t_e = model_calc(neg_t_e, neg_t_t, neg_r_t)
        
        if config_dict['L1_flag']:
            pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - pos_t_e), 1, keep_dims = True)
            neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - neg_t_e), 1, keep_dims = True)
            predict = pos
        else:
            pos = tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
            neg = tf.reduce_sum((neg_h_e + neg_r_e - neg_t_e) ** 2, 1, keep_dims = True)
            predict = pos

        with tf.name_scope("output"):
            loss = tf.reduce_sum(tf.maximum(pos - neg + margin, 0))

        return {
            'pos_h': pos_h,
            'pos_t': pos_t,
            'pos_r': pos_r,

            'neg_h': neg_h,
            'neg_t': neg_t,
            'neg_r': neg_r,
            
            'ent_embedding': ent_embeddings,
            'rel_embeddings': rel_embeddings,
            'ent_transfer': ent_transfer,
            'rel_transfer': rel_transfer,

            'predict': predict,
            'loss': loss
            }

def main(argv):
	config_dict = init_config(is_test=True, load_curr_model=True)
	if (config_dict['test_flag']):
		test_lib.init()
		config_dict['relation'] = test_lib.get_relation_total()
		config_dict['entity'] = test_lib.get_entity_total()
		config_dict['batch'] = test_lib.get_entity_total()
		config_dict['batch_size'] = config_dict['batch']
	else:
		lib.init()
		config_dict['relation'] = lib.get_relation_total()
		config_dict['entity'] = lib.get_entity_total()
		config_dict['batch_size'] = lib.get_triple_total() // config_dict['nbatches']

	with tf.Graph().as_default():
		sess = tf.Session()
		with sess.as_default():
			initializer = tf.contrib.layers.xavier_initializer(uniform = False)
			with tf.variable_scope("model", reuse=None, initializer = initializer):
				train_model = init_model(config_dict=config_dict)

			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.GradientDescentOptimizer(0.001)
			grads_and_vars = optimizer.compute_gradients(train_model['loss'])
			train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
			saver = tf.train.Saver()
			sess.run(tf.global_variables_initializer())
			if (config_dict['load_from_data']):
				saver.restore(sess, './model.vec')

			def train_step(pos_h_batch, pos_t_batch, pos_r_batch, neg_h_batch, neg_t_batch, neg_r_batch):
				feed_dict = {
					train_model['pos_h']: pos_h_batch,
					train_model['pos_t']: pos_t_batch,
					train_model['pos_r']: pos_r_batch,
					train_model['neg_h']: neg_h_batch,
					train_model['neg_t']: neg_t_batch,
					train_model['neg_r']: neg_r_batch
				}
				_, step, loss = sess.run(
					[train_op, global_step, train_model['loss']], feed_dict)
				return(loss)

			def test_step(pos_h_batch, pos_t_batch, pos_r_batch):
				feed_dict = {
					train_model['pos_h']: pos_h_batch,
					train_model['pos_t']: pos_t_batch,
					train_model['pos_r']: pos_r_batch,
				}
				step, predict = sess.run(
					[global_step, train_model['predict']], feed_dict)
				return(predict)

			ph = np.zeros(config_dict['batch_size'], dtype = np.int32)
			pt = np.zeros(config_dict['batch_size'], dtype = np.int32)
			pr = np.zeros(config_dict['batch_size'], dtype = np.int32)
			nh = np.zeros(config_dict['batch_size'], dtype = np.int32)
			nt = np.zeros(config_dict['batch_size'], dtype = np.int32)
			nr = np.zeros(config_dict['batch_size'], dtype = np.int32)

			ph_addr = ph.__array_interface__['data'][0]
			pt_addr = pt.__array_interface__['data'][0]
			pr_addr = pr.__array_interface__['data'][0]
			nh_addr = nh.__array_interface__['data'][0]
			nt_addr = nt.__array_interface__['data'][0]
			nr_addr = nr.__array_interface__['data'][0]

			lib.get_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, 
                ctypes.c_void_p, ctypes.c_void_p, 
                ctypes.c_void_p, ctypes.c_void_p, 
                ctypes.c_int
                ]
			test_lib.get_head_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
                ]
			test_lib.get_tail_batch.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p
                ]
			test_lib.test_head.argtypes = [ctypes.c_void_p]
			test_lib.test_tail.argtypes = [ctypes.c_void_p]

			if not config_dict['test_flag']:
				for times in range(config_dict['train_times']):
					res = 0.0
					for batch in range(config_dict['nbatches']):
						lib.get_batch(ph_addr, pt_addr, pr_addr, nh_addr, nt_addr, nr_addr, config_dict['batch_size'])
						res += train_step(ph, pt, pr, nh, nt, nr)
						current_step = tf.train.global_step(sess, global_step)
					print(times)
					print(res)
				saver.save(sess, './model.vec')
			else:
				total = test_lib.get_test_total()
				for times in range(total):
					test_lib.get_head_batch(ph_addr, pt_addr, pr_addr)
					res = test_step(ph, pt, pr)
					test_lib.test_head(res.__array_interface__['data'][0])

					test_lib.get_tail_batch(ph_addr, pt_addr, pr_addr)
					res = test_step(ph, pt, pr)
					test_lib.test_tail(res.__array_interface__['data'][0])
					print(times)
					if (times % 50 == 0):
						test_lib.test()
				test_lib.test()

if __name__ == "__main__":
	tf.app.run()

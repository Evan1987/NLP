"""
coding: utf-8
@author: Cigar
Tensorflow implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling"((https://arxiv.org/abs/1609.01454))
https://github.com/applenob/RNN-for-Joint-NLU
"""

import tensorflow as tf
import numpy as np
class Model:
    
    def __init__(self, input_length, embedding_size, hidden_size, 
                 vocab_size, slot_size, intent_size, batch_size=16):
        self.input_length = input_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.slot_size = slot_size
        self.intent_size = intent_size
        self.batch_size = batch_size
        self._graph = tf.Graph()
        
    @property
    def graph(self):
        tf.reset_default_graph()
        with self._graph.as_default():
            self.encoder_inputs = tf.placeholder(dtype=tf.int32, 
                                                 shape=[self.input_length, self.batch_size], 
                                                 name="encoder_inputs")
            self.encoder_inputs_actual_length = tf.placeholder(dtype=tf.int32, 
                                                               shape=[self.batch_size], 
                                                               name="encoder_inputs_actual_length")
            self.decoder_targets = tf.placeholder(dtype=tf.int32, 
                                                  shape=[self.batch_size, self.input_length], 
                                                  name="decoder_targets")
            self.intent_targets = tf.placeholder(dtype=tf.int32, 
                                                 shape=[self.batch_size], 
                                                 name="intent_targets")
            self.embeddings = tf.Variable(tf.random_uniform(shape=[self.vocab_size, self.embedding_size], 
                                                            minval=-0.1, 
                                                            maxval=0.1), 
                                          dtype=tf.float32, 
                                          name="embedding")
            self.encoder_inputs_embedded = tf.nn.embedding_lookup(self.embeddings, self.encoder_inputs)
            
            '''encoder'''
            encoder_fw_cell_0 = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size)  # forward
            encoder_bw_cell_0 = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size)  # backward
            encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(cell=encoder_fw_cell_0, output_keep_prob=0.5)
            encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(cell=encoder_bw_cell_0, output_keep_prob=0.5)
            
            # Return (outputs, output_states)
            # outputs: (output_fw, output_bw)
            # if time_major == True: (which influences the major axis of output and input required)
            # output_fw: [max_time, batch_size, cell_fw_num]
            # output_bw: [max_time, batch_size, cell_bw_num]
            #
            # output_states: (output_state_fw, output_state_bw)
            # output_state_fw.h or c: [batch_size, cell_fw_num]
            # output_state_bw.h or c: [batch_size, cell_bw_num]
            (encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state) =\
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_fw_cell,
                                                cell_bw=encoder_bw_cell, 
                                                inputs=self.encoder_inputs_embedded, 
                                                sequence_length=self.encoder_inputs_actual_length, 
                                                dtype=tf.float32, 
                                                time_major=True)
            
            # shape: [max_time, batch_size, cell_fw_num + cell_bw_num] => [max_time, batch_size, hidden_size * 2]
            encoder_outputs = tf.concat([encoder_fw_outputs, encoder_bw_outputs], axis=2)
            # shape: [batch_size, cell_fw_num + cell_bw_num] => [batch_size, hidden_size * 2]
            encoder_final_state_c = tf.concat([encoder_fw_final_state.c, encoder_bw_final_state.c], axis=1)
            # shape: [batch_size, cell_fw_num + cell_bw_num] => [batch_size, hidden_size * 2]
            encoder_final_state_h = tf.concat([encoder_fw_final_state.h, encoder_bw_final_state.h], axis=1)
            self.encoder_final_state = tf.contrib.rnn.LSTMStateTuple(
                c=encoder_final_state_c, 
                h=encoder_final_state_h
            )
            print("encoder outputs: ", encoder_outputs)
            print("encoder outputs[0]: ", encoder_outputs[0])
            print("encoder final state c: ", encoder_final_state_c)
            
            
            '''decoder'''
            decoder_lengths = self.encoder_inputs_actual_length
            self.slot_W = tf.Variable(tf.random_uniform(shape=[self.hidden_size * 2, self.slot_size], 
                                                        minval=-1.0, 
                                                        maxval=1.0), 
                                      dtype=tf.float32, 
                                      name="slot_w")
            self.slot_b = tf.Variable(tf.zeros([1, self.slot_size]), dtype=tf.float32, name="slot_b")
            intent_W = tf.Variable(tf.random_uniform(shape=[self.hidden_size * 2, self.intent_size], 
                                                     minval=-0.1, 
                                                     maxval=1.0), 
                                   dtype=tf.float32, 
                                   name="intent_w")
            intent_b = tf.Variable(tf.zeros([1, self.intent_size]), dtype=tf.float32, name="intent_b")
            
            # shape: [batch_size, intent_size]
            intent_logits = encoder_final_state_h @ intent_W + intent_b
            # shape: [batch_size, 1]
            self.intent = tf.argmax(intent_logits, axis=1)
            
            def initial_fn():
                """
                customize sampling for first iter
                :return: (finished, next_inputs)
                """
                initial_elements_finished = (0 >= decoder_lengths)  # all false
                # shape: [batch_size, embedding_size]
                sos_step_embedded = tf.nn.embedding_lookup(self.embeddings,
                                                           tf.constant(value=2, shape=[self.batch_size], dtype=tf.int32))
                
                # sos_step_embedded: [batch_size, embedding_size]
                # encoder_outputs[0]: [batch_size, hidden_size * 2]
                # initial_input: [batch_size, hidden_size * 2 + embedding_size]
                inital_input = tf.concat([sos_step_embedded, encoder_outputs[0]], axis=1)
                return initial_elements_finished, inital_input
            
            def sample_fn(time, outputs, state):
                """
                takes (time, outputs, state)
                :return: sample_ids
                """
                # 选择logit最大的下标作为sample
                prediction_id = tf.to_int32(tf.argmax(outputs, axis=1))
                return prediction_id
            
            def next_inputs_fn(time, outputs, state, sample_ids):
                """
                callable that takes `(time, outputs, state, sample_ids)`
                :return: (finished, next_inputs, next_state)
                """
                # 上一个时间节点上的输出类别，获取embedding再作为下一个时间节点的输入
                pred_embedding = tf.nn.embedding_lookup(self.embeddings, sample_ids)
                
                pad_step_embedded = tf.zeros(shape=[self.batch_size, self.hidden_size * 2 + self.embedding_size],
                                             dtype=tf.float32)
                
                # pred_embedding: [batch_size, embedding_size]
                # encoder_outputs[time]: [batch_size, hidden_size * 2]
                # next_input: [batch_size, embedding_size + hidden_size * 2]
                next_input = tf.concat([pred_embedding, encoder_outputs[time]], axis=1)
                
                #[batch_size] bool
                elements_finished = (time >= decoder_lengths)
                # scalar bool
                all_finished = tf.reduce_all(elements_finished)  
                
                # like ifelse in R or switch. tf.cond(bool, true_fn, false_fn)
                # 如果已经全部完成则输入pad值（all 0 tensor），否则进行下一步输入
                next_inputs = tf.cond(all_finished, lambda: pad_step_embedded, lambda: next_input)
                next_state = state
                return elements_finished, next_inputs, next_state
            
            my_helper = tf.contrib.seq2seq.CustomHelper(initial_fn, sample_fn, next_inputs_fn)
            
                
            def decode(helper, scope, reuse=None):
                with tf.variable_scope(scope, reuse=reuse):
                    # [max_time, batch_size, hidden_size * 2] => [batch_size, max_time, hidden_size * 2]
                    memory = tf.transpose(encoder_outputs, [1, 0, 2])
                    attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                        num_units=self.hidden_size,
                        memory=memory,
                        memory_sequence_length=self.encoder_inputs_actual_length
                    )
                    
                    cell = tf.contrib.rnn.LSTMCell(num_units=self.hidden_size * 2)
                    
                    attn_cell = tf.contrib.seq2seq.AttentionWrapper(
                        cell=cell, 
                        attention_mechanism=attention_mechanism,
                        attention_layer_size=self.hidden_size
                    )
                    
                    out_cell = tf.contrib.rnn.OutputProjectionWrapper(
                        cell=attn_cell, 
                        output_size=self.slot_size,
                        reuse=reuse
                    )
                    
                    decoder = tf.contrib.seq2seq.BasicDecoder(
                        cell=out_cell,
                        helper=helper,
                        initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=self.batch_size)
                    )
                    
                    final_outputs, final_state, final_sequence_lengths =\
                        tf.contrib.seq2seq.dynamic_decode(
                            decoder=decoder, 
                            output_time_major=True, 
                            impute_finished=True, 
                            maximum_iterations=self.input_length
                        )
                    
                    return final_outputs
            
            outputs = decode(helper=my_helper, scope="decode", reuse=None)
            print("outputs: ", outputs)
            print("outputs.rnn_output: ", outputs.rnn_output)
            print("outputs.sample_id: ", outputs.sample_id)
            self.decoder_prediction = outputs.sample_id
            
            decoder_max_steps, decoder_batch_size, decoder_dim =\
                tf.unstack(tf.shape(outputs.rnn_output))
            
            # decoder_targets: [batch_size, input_length]
            # decoder_targets_time_majored: [input_length, batch_size]
            self.decoder_targets_time_majored = tf.transpose(self.decoder_targets, [1, 0])
            
            # shape: [decoder_max_steps, batch_size]
            self.decoder_targets_true_length = self.decoder_targets_time_majored[: decoder_max_steps]
            print("decoder targets true length: ", self.decoder_targets_true_length)
            
            # 定义mask，使 padding不计入loss
            self.mask = tf.to_float(tf.not_equal(self.decoder_targets_true_length, 0))
            # 定义 slot标注损失
            slot_loss = tf.contrib.seq2seq.sequence_loss(
                logits=outputs.rnn_output,
                targets=self.decoder_targets_true_length,
                weights=self.mask
            )
            # 定义 intent分类损失
            intent_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                labels=tf.one_hot(self.intent_targets, depth=self.intent_size, dtype=tf.float32),
                logits=intent_logits
            ))
            
            self.loss = slot_loss + intent_loss
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001, name="a_optimizer")
            self.grads, self.vars = zip(*optimizer.compute_gradients(self.loss))
            print("vars for loss function: ", self.vars)
            self.gradients, _ = tf.clip_by_global_norm(self.grads, 5)
            self.train_op = optimizer.apply_gradients(zip(self.gradients, self.vars))

            self.saver = tf.train.Saver()
        
        return self._graph
        
    def step(self, sess, mode, indexed_batch):
        """calculate with batch"""
        if mode not in ["train", "test"]:
            raise Exception("Invalid mode!")

        sin, true_length, sout, intent = list(zip(*indexed_batch))
        sin, true_length, sout, intent = map(list, [sin, true_length, sout, intent])
        if mode == "train":
            outfeeds = [self.train_op, self.loss, self.decoder_prediction,
                        self.intent, self.mask, self.slot_W]
            feed_dict = {self.encoder_inputs: np.transpose(sin, [1, 0]),
                         self.encoder_inputs_actual_length: true_length,
                         self.decoder_targets: sout,
                         self.intent_targets: intent}
        elif mode == "test":
            outfeeds = [self.decoder_prediction, self.intent]
            feed_dict = {self.encoder_inputs: np.transpose(sin, [1, 0]),
                         self.encoder_inputs_actual_length: true_length}

        results = sess.run(outfeeds, feed_dict=feed_dict)
        return results


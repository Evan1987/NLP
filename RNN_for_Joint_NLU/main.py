
"""
coding: utf-8
@date: 2018-12-27
@author: Cigar
Tensorflow implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling"
((https://arxiv.org/abs/1609.01454))
https://github.com/applenob/RNN-for-Joint-NLU
"""

import tensorflow as tf
import numpy as np
import _Data
import _Model 
import _Metric
import time
import random
from _utils import u_constant
path = u_constant.PATH_ROOT + "for learn/Python/RNN-for-Joint-NLU/"
train_data_path = path + "dataset/atis-2.train.w-intent.iob"
test_data_path = path + "dataset/atis-2.dev.w-intent.iob"
model_path = path + "model/"

data_info = _Data.DataUtils(train_data_path, test_data_path, length=50)
indexed_train_data = data_info.to_index(data_info.train_data)  # 4478
indexed_test_data = data_info.to_index(data_info.test_data)  # 500

input_length = data_info.length  # 50
embedding_size = 64
hidden_size = 100
n_layers = 2
batch_size = 16
vocab_size = len(data_info.word2index)  # 871
slot_size = len(data_info.tag2index)  # 124
intent_size = len(data_info.intent2index)  # 22
epoch_num = 60



def get_iternum_from_ckpt(ckpt_path):
    iters = ckpt_path.split(".ckpt-")[1]
    return int(iters)

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu80%的显存
config.gpu_options.allow_growth = True      #程序按需申请内存

model = _Model.Model(input_length, embedding_size, hidden_size, vocab_size, 
                     slot_size, intent_size, batch_size)
with tf.Session(graph=model.graph, config=config) as sess:
    ckpt = tf.train.get_checkpoint_state(model_path)
    if ckpt is not None:
        ckpt_path = ckpt.model_checkpoint_path
        model.saver.restore(sess, model_path)
        iteration = get_iternum_from_ckpt(ckpt_path)
    else:
        sess.run(tf.global_variables_initializer())
        iteration = 1
    start = time.time()
    avg_loss = 0.0
    for epoch in range(epoch_num):
        for batch in data_info.get_batch(batch_size, indexed_train_data):
            # 执行一个 batch 的训练``````````````
            _, loss, decoder_prediction, intent, mask = model.step(sess, "train", batch)
            avg_loss += loss
            if iteration % 100 == 0:
                end = time.time()
                print("Iteration: %d " % iteration, 
                      "Avg. Train_loss: %.4f" % (avg_loss / 100),
                      "%.4f sec / batch" % ((end - start) / 100))
                start = time.time()
                avg_loss = 0.0
            iteration += 1
        # 每个 epoch保存一次模型
        model.saver.save(sess, model_path + "/seq2seq.ckpt", global_step=iteration)
        
        # 测试
        pred_slots = []
        true_slots = []
        slot_accs = []
        intent_accs = []
        for j, batch in enumerate(data_info.get_batch(batch_size, indexed_test_data)):
            decoder_prediction, intent = model.step(sess, "test", batch)
            decoder_prediction = np.transpose(decoder_prediction, [1, 0])
        
            test_in, test_true_length, test_out, test_intent = map(np.asarray, list(zip(*batch)))
            
            if j == 0:
                # 随机选出一个样本打印
                index = random.choice(range(len(batch)))
                print("---Epoch Summary for epoch %d---" % (epoch + 1))
                print("Input Sentence   :", data_info.recover_indexed_seq(test_in[index], "word")[: test_true_length[index]])
                print("Slot Truth       :", data_info.recover_indexed_seq(test_out[index], "slot")[: test_true_length[index]])
                print("Slot Pred        :", data_info.recover_indexed_seq(decoder_prediction[index], "slot")[: test_true_length[index]])
                print("Intent Truth     :", data_info.index2intent[test_intent[index]])
                print("Intent Pred      :", data_info.index2intent[intent[index]])
        
            slot_pred_length = np.shape(decoder_prediction)[1]
        
            # 确保输出与输入长度一致，便于指标计算
            # pad_width: pad element num for each axis
            # ((before_0, after_0), (before_1, after_1), ..., (before_N, after_N))
            pred_padded = np.lib.pad(decoder_prediction, 
                                     pad_width=((0, 0), (0, input_length - slot_pred_length)), 
                                     mode="constant", 
                                     constant_values=0)
            
            pred_slots.append(pred_padded)
            true_slots.append(test_out)
            
            slot_acc = _Metric.acc_for_sequence_batch(test_out, pred_padded, padding_token=0)
            intent_acc = _Metric._acc(test_intent, intent)
            
            slot_accs.append(slot_acc)
            intent_accs.append(intent_acc)
        
        pred_slots = np.vstack(pred_slots)
        true_slots = np.vstack(true_slots)
        test_slot_f1 = _Metric.f1_for_sequence_batch(true_slots, pred_slots, padding_token=0)
        
        print("Test Slot Avg.Acc   : %.4f" % np.average(slot_accs))
        print("Test Slot f1        : %.4f" % test_slot_f1)
        print("Test Intent Avg. Acc: %.4f" % np.average(intent_accs))
        


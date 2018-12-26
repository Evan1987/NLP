"""
coding: utf-8
@author: Cigar
Tensorflow implementation of "Attention-Based Recurrent Neural Network Models for Joint Intent Detection and Slot Filling"
((https://arxiv.org/abs/1609.01454))
https://github.com/applenob/RNN-for-Joint-NLU
"""

import itertools
import random

class DataUtils:
    
    def __init__(self, train_data_path, test_data_path, length=50):
        self.train_data = self.load_data(train_data_path, length)
        self.test_data = self.load_data(test_data_path, length)
        self.length = length
        self.word2index, self.index2word, self.tag2index,\
        self.index2tag, self.intent2index, self.index2intent = self.get_info_from_training(self.train_data)
    
    def treat_line(self, line, length):
        """处理读入数据的每一行，并根据序列长度进行padding
        原始数据结构: 原始句子\t标注序列 intent ，如下

        'BOS i want to fly from baltimore to dallas round trip EOS
        \tO O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'

        分割成[原始句子的词，标注的序列，intent]
        :param line: 原始文件中的一行
        :param length: 序列长度
        :return: 返回指定length的原始序列、标注序列以及intent
        """
        line = line.strip()
        text, tag = line.split("\t")
        seq_in = text.split()[1: -1]  # 去掉原始序列的 BOS和 EOS
        seq_out = tag.split()  
        intent = seq_out[-1]
        seq_out = seq_out[1: -1]  # 去掉标注序列的 BOS对应位置, EOS对应的是intent

        seq_length = len(seq_in)
        assert len(seq_out) == seq_length,\
            "labeled seq length %d doesn't match original text! expect %d" % (len(seq_out), seq_length)

        # padding，原始序列和标注序列结尾+<EOS>+n×<PAD>
        if seq_length < length:
            append_seq = list(itertools.repeat("<PAD>", length - seq_length))
            append_seq[0] = "<EOS>"
            seq_in.extend(append_seq)
            seq_out.extend(append_seq)
        else:
            seq_in = seq_in[:length]
            seq_in[-1] = "<EOS>"
            seq_out = seq_out[:length]
            seq_out[-1] = "<EOS>"
        return seq_in, seq_out, intent 
    
    def load_data(self, data_path, length):
        """读入文件并解析每行，每行处理成指定length的原始序列、标记序列以及intent
        """
        data = []
        with open(data_path, "r") as f:
            for line in f:
                line = line.strip()
                seq_in, seq_out, intent = self.treat_line(line, length)
                data.append((seq_in, seq_out, intent))
            f.close()
        return data
    
    def __get_index_map(self, init_map, iters):
        """给定初始索引，对于不在索引中的key进行添加，返回索引和反索引
        """
        now_length = len(init_map)
        for ele in iters:
            try:
                init_map[ele]
            except KeyError:
                init_map[ele] = now_length
                now_length += 1
        reverse_map = {v: k for k, v in init_map.items()}
        return init_map, reverse_map
               
    def get_info_from_training(self, train_data):
        """根据训练集生成索引数据和反索引数据
        :return: word索引、slot索引、intent索引及各自的反索引
        """
        seq_in, seq_out, intent = list(zip(*train_data))
        vocab = dict.fromkeys(itertools.chain.from_iterable(seq_in))
        slot_tag = dict.fromkeys(itertools.chain.from_iterable(seq_out))
        intent_tag = dict.fromkeys(intent)
        # 生成word2index和 index2word
        word2index, index2word =\
            self.__get_index_map({"<PAD>": 0, "<UNK>": 1, "<SOS>": 2, "<EOS>": 3}, vocab)
        
        # 生成tag2index 和 index2tag
        tag2index, index2tag =\
            self.__get_index_map({"<PAD>": 0, "<UNK>": 1, "0": 2}, slot_tag)
        
        # 生成intent2index 和 index2intent
        intent2index, index2intent =\
            self.__get_index_map({"<UNK>": 0}, intent_tag)
        
        return word2index, index2word, tag2index, index2tag, intent2index, index2intent
    
    def get_batch(self, batch_size):
        """生成训练batch
        """
        random.shuffle(self.train_data)
        index = 0
        while index + batch_size < len(self.train_data):
            batch = self.train_data[index: (index + batch_size)]
            index += batch_size
            yield batch
    
    def to_index(self, train):
        """文本索引化
        """
        new_train = []
        word2index_f = lambda x: self.word2index.get(x, self.word2index["<UNK>"])
        slot2index_f = lambda x: self.tag2index.get(x, self.tag2index["<UNK>"])
        intent2index_f = lambda x: self.intent2index.get(x, self.intent2index["<UNK>"])
        for sin, sout, intent in train:
            sin_ix = [word2index_f(x) for x in sin]
            true_length = sin.index("<EOS>")
            sout_ix = [slot2index_f(x) for x in sout]
            intent_ix = intent2index_f(intent)
            new_train.append((sin_ix, true_length, sout_ix, intent_ix))
        return new_train

    def recover_indexed_seq(self, indexed_seq, index_type="word"):
        """
        将编码索引后的序列反索引化
        :param indexed_seq: seq of index int
        :param index_type: which kind of index
        :return: recovered seq
        """
        index_type = index_type.lower()
        if index_type in ["word", "words", "w"]:
            reverse_map = self.index2word
        elif index_type in ["slot", "slots", "tag", "s", "t", "tags"]:
            reverse_map = self.index2tag
        elif index_type in ["intent", "intents", "i"]:
            reverse_map = self.index2intent
        else:
            raise Exception("Invalid index_type! Possible choice are 'word', 'slot', 'intent'!")

        return [reverse_map[x] for x in indexed_seq]
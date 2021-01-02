
import os
import pandas as pd
import numpy as np
import jieba
import re
from sklearn.utils import shuffle
from typing import *


path = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(path, "data")
vocab_file = os.path.join(data_path, "vocab.txt")
word_vocab_file = os.path.join(data_path, "word_vocab.tsv")
train_file = os.path.join(data_path, "train.csv")
test_file = os.path.join(data_path, "test.csv")
eval_file = os.path.join(data_path, "dev.csv")


class Vocab(object):

    def __init__(self, file: str):
        self.file = file
        self._word_index_mapping: Dict[str, int] = {}
        self._index_word_mapping: Dict[int, str] = {}

    def initialize(self):
        if not os.path.exists(self.file):
            raise IOError(f"{self.file} not exists!")
        with open(self.file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if line := line.strip():
                    self._word_index_mapping[line] = i
                    self._index_word_mapping[i] = line

    def __len__(self):
        return len(self.word_index_mapping)

    @property
    def vocab_size(self):
        return max(self.index_word_mapping) + 1

    @property
    def word_index_mapping(self) -> Dict[str, int]:
        if not self._word_index_mapping:
            self.initialize()
        return self._word_index_mapping

    @property
    def index_word_mapping(self) -> Dict[int, str]:
        if not self._index_word_mapping:
            self.initialize()
        return self._index_word_mapping

    def contains_word(self, word: str):
        return word in self.word_index_mapping

    def get_word_index(self, word: str) -> Optional[int]:
        return self.word_index_mapping.get(word)

    def get_sentence_indexes(self, sentence: List[str], ignore_na: bool = False) -> List[Optional[int]]:
        res = []
        for w in sentence:
            if (index := self.get_word_index(w)) is None and ignore_na:
                continue
            res.append(index)
        return res


def word2vec(word: str, length: int = None, model=None) -> np.ndarray:
    """静态 word2vec"""
    if model is None:
        if length is None or length <= 0:
            raise ValueError("length is not positive when model is not supplied")
        return np.zeros(length)
    return model.wv[word]


def pad_sequence(seq: List[int], max_length: int) -> np.ndarray:
    seq = np.array(seq)
    if max_length > 0:
        if len(seq) < max_length:
            return np.pad(seq, (0, max_length - len(seq)))
        elif len(seq) > max_length:
            return seq[: max_length]
    return seq


def pad_sequences(sequences: List[List], max_length: int) -> np.ndarray:
    if max_length < 0:
        max_length = max(len(seq) for seq in sequences)
    res = np.zeros(shape=(len(sequences), max_length))
    for i, seq in enumerate(sequences):
        n = min(len(seq), max_length)
        res[i, :n] = seq[:n]
    return res


CHAR_VOCAB = Vocab(vocab_file)
WORD_VOCAB = Vocab(word_vocab_file)


def load_char_data(csv_file: str, size: int = -1) -> Tuple[List[List], List[List], List[int]]:
    df = pd.read_csv(csv_file)
    p = df["sentence1"].values
    h = df["sentence2"].values
    label = df["label"].values
    if size > 0:
        p, h, label = map(lambda x: x[:size], [p, h, label])
    
    p, h, label = shuffle(p, h, label.tolist())
    p_index_list = [CHAR_VOCAB.get_sentence_indexes(content.lower(), ignore_na=True) for content in p]
    h_index_list = [CHAR_VOCAB.get_sentence_indexes(content.lower(), ignore_na=True) for content in h]
    return p_index_list, h_index_list, label

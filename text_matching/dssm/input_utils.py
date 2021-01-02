
import numpy as np
import tensorflow as tf
from utils.data_utils import load_char_data, pad_sequence
from typing import *


class DataGenerator(object):

    def __init__(self, max_length: int):
        self.max_length = max_length
        self.p_indexes, self.h_indexes, self.labels = [], [], []

    def fit(self, file: str):
        self.p_indexes, self.h_indexes, self.labels = load_char_data(file)
        return self

    def __len__(self):
        return len(self.p_indexes)

    def get_data_iterator(self) -> Iterator[Tuple[Dict, int]]:
        for p, h, y in zip(self.p_indexes, self.h_indexes, self.labels):
            yield (pad_sequence(p, self.max_length), pad_sequence(h, self.max_length)), np.array([y])

    def input_fn(self, batch_size: int, repeat: int = 1, prefetch: int = 2) -> tf.data.Dataset:
        output_shapes = (tf.TensorShape([None]), tf.TensorShape([None])), tf.TensorShape([1])
        dataset = tf.data.Dataset.from_generator(
            self.get_data_iterator,
            output_types=((tf.int32, tf.int32), tf.int32),
            output_shapes=output_shapes
        ).repeat(repeat)
        
        if self.max_length < 0:
            dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=output_shapes, drop_remainder=False)
        else:
            dataset = dataset.batch(batch_size=batch_size, drop_remainder=False)

        return dataset.prefetch(buffer_size=prefetch)





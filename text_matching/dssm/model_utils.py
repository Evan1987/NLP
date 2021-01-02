
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
from evan_utils.utensorflow.model import KerasModel
from typing import *


class DSSM(KerasModel):
    """Deep-Semantic-Similarity-Model"""

    def __init__(self, input_length: int, learning_rate: float, vocab_size: int, vocab_embedding_length: int, keep_prob: float):
        self.input_length = input_length
        self.lr = learning_rate
        self.vocab_size = vocab_size
        self.vocab_embedding_length = vocab_embedding_length
        self.keep_prob = keep_prob

        p = layers.Input(shape=[self.input_length], name="p", dtype="int32")
        h = layers.Input(shape=[self.input_length], name="h", dtype="int32")
        self.inputs = [p, h]
        self.embedding = layers.Embedding(input_dim=self.vocab_size, output_dim=self.vocab_embedding_length)
        p_embed = self.embedding(p)
        h_embed = self.embedding(h)

        p_context = DSSM.fully_connect(p_embed, 1 - self.keep_prob)
        h_context = DSSM.fully_connect(h_embed, 1 - self.keep_prob)
        # cosine similarity [-1, 1] to be prob with softmax: e^x / e^x + e^(1 - x) -> 1 / (1 + e^(-(2x-1)))
        # logit for sigmoid 2x - 1
        similarity = layers.dot([p_context, h_context], normalize=True, axes=1)
        prob = layers.Lambda(lambda x: 2.0 * x - 1.0)(similarity)

        self._model = Model(self.inputs, prob)
        self._model.compile(optimizer=Adam(lr=self.lr), metrics=["accuracy"],
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    @staticmethod
    def fully_connect(x: tf.Tensor, drop_prob: float) -> tf.Tensor:
        x = layers.Dense(256, activation="tanh")(x)
        x = layers.Dropout(drop_prob)(x)
        x = layers.Dense(512, activation="tanh")(x)
        x = layers.Dropout(drop_prob)(x)
        x = layers.Dense(256, activation="tanh")(x)
        x = layers.Dropout(drop_prob)(x)
        return layers.Flatten()(x)

    @property
    def model(self) -> Model:
        return self._model

    def train(self, train_data: tf.data.Dataset, test_data=None, epochs: int = 10, validation_steps: int = 100,
              callbacks=None):
        h = self.model.fit(train_data, validation_data=test_data, epochs=epochs, verbose=1, callbacks=callbacks,
                           validation_steps=validation_steps, use_multiprocessing=True, workers=4)
        return h

    def test(self, *, test_data: tf.data.Dataset):
        loss = 0
        for inputs in test_data:
            loss += self.model.predict(inputs)
        return loss

    def predict(self, inputs, **kwargs):
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Require inputs have {len(self.inputs)} keys, got {len(inputs)}")
        return self.model(inputs)





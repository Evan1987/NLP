
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers, Model, losses
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from evan_utils.utensorflow.model import KerasModel
from typing import *


class ConvNet(KerasModel):

    def __init__(self, input_length: int, learning_rate: float, vocab_size: int, vocab_embedding_length: int,
                 keep_prob: float, n_filter: int, filter_height: int, filter_width: int):
        self.input_length = input_length
        self.lr = learning_rate
        self.vocab_size = vocab_size
        self.vocab_embedding_length = vocab_embedding_length
        self.keep_prob = keep_prob
        self.n_filter = n_filter
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.kernel_size = (self.filter_height, self.filter_width)

        p = layers.Input(shape=[self.input_length], name="p", dtype="int32")
        h = layers.Input(shape=[self.input_length], name="h", dtype="int32")
        self.inputs = [p, h]
        self.embedding = layers.Embedding(input_dim=self.vocab_size, output_dim=self.vocab_embedding_length)
        p_embed = self.embedding(p)                                         # [input_length, embedding_length]
        h_embed = self.embedding(h)                                         # [input_length, embedding_length]

        p_sim = ConvNet.cnn(p_embed, self.n_filter, self.kernel_size)       # [n_filter]
        h_sim = ConvNet.cnn(h_embed, self.n_filter, self.kernel_size)       # [n_filter]

        sim = layers.Dense(units=self.n_filter, use_bias=False,
                           kernel_initializer=RandomNormal(mean=0.0, stddev=1.0))(p_sim)
        sim = layers.Dot([h_sim, sim], axes=1)

        x = layers.Concatenate()([p_sim, sim, h_sim])
        x = layers.Dense(units=256, activation="relu")(x)
        x = layers.Dropout(1.0 - self.keep_prob)(x)
        logits = layers.Dense(units=2, activation="relu")(x)
        self._model = Model(self.inputs, logits)
        self._model.compile(optimizer=Adam(lr=self.lr), loss=losses.CategoricalCrossentropy(from_logits=True))

    @staticmethod
    def cnn(x: tf.Tensor, n_filter: int, kernel_size: Tuple[int, int]) -> tf.Tensor:
        x = layers.Lambda(lambda v: K.expand_dims(K.permute_dimensions(v, (0, 2, 1)), axis=3))(x)
        x = layers.Conv2D(n_filter, kernel_size, activation="relu", data_format="channel_last")(x)
        x = layers.GlobalMaxPool2D(data_format="channel_last")(x)
        return x

    @property
    def model(self) -> Model:
        return self._model

    def train(self, train_data: tf.data.Dataset, test_data=None, epochs: int = 10, validation_steps: int = 100,
              callbacks=None):
        h = self.model.fit(train_data, validation_data=test_data, epochs=epochs, verbose=1, callbacks=callbacks,
                           validation_steps=validation_steps, use_multiprocessing=True, workers=4)
        return h

    def predict(self, inputs, **kwargs):
        if len(inputs) != len(self.inputs):
            raise ValueError(f"Require inputs have {len(self.inputs)} keys, got {len(inputs)}")
        return self.model(inputs)



import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint
from utils.data_utils import CHAR_VOCAB, train_file, eval_file, test_file, path
from dssm.input_utils import DataGenerator
from dssm.model_utils import DSSM
from typing import *


MAX_SEQ_LENGTH = 15
CHAR_EMBEDDING_SIZE = 100
LEARNING_RATE = 2e-3
KEEP_PROB = 0.8
EPOCHS = 50
BATCH_SIZE = 128

MODEL_DIR = os.path.join(path, "dssm/model")
FINAL_MODEL_FILE = "model.h5"
MODEL_FILE = os.path.join(MODEL_DIR, FINAL_MODEL_FILE)


def get_callbacks() -> List[tf.keras.callbacks.Callback]:
    checkpoint_file = os.path.join(MODEL_DIR, "model_{epoch:02d}-{val_loss:.4f}.hdf5")
    checkpoints = ModelCheckpoint(
        filepath=checkpoint_file, monitor="val_loss", save_weights_only=False, verbose=1, save_best_only=True)
    # board = TensorBoard(log_dir=get_board_log_path("geo"), batch_size=BATCH_SIZE, profile_batch=0)
    # os.makedirs(board.log_dir, exist_ok=True)
    return [checkpoints]


train_data = DataGenerator(MAX_SEQ_LENGTH).fit(train_file)
train_batches = train_data.input_fn(BATCH_SIZE).shuffle(buffer_size=2 * BATCH_SIZE, seed=0)
eval_data = DataGenerator(MAX_SEQ_LENGTH).fit(eval_file)
test_data = DataGenerator(MAX_SEQ_LENGTH).fit(test_file)

model = DSSM(MAX_SEQ_LENGTH, LEARNING_RATE, CHAR_VOCAB.vocab_size, CHAR_EMBEDDING_SIZE, KEEP_PROB)
model.summary()

if os.path.exists(MODEL_FILE):
    print("Load model ...")
    model.load_model(MODEL_FILE)

model.train(train_data=train_batches,
            test_data=eval_data.input_fn(100), epochs=EPOCHS, callbacks=get_callbacks())
model.save_model(MODEL_FILE)


model.load_model(MODEL_FILE)

similarities = []
labels = []
for test_inputs, test_labels in test_data.input_fn(batch_size=100):
    similarities.append(model.predict(test_inputs))
    labels.append(test_labels)

similarities = np.vstack(similarities)
labels = np.vstack(labels)
acc = ((similarities > 0.5).astype(int) == labels).mean()
loss = K.binary_crossentropy(labels.astype(np.float32), similarities, from_logits=True).numpy().mean()

print("Acc: ", acc, " Loss: ", loss)

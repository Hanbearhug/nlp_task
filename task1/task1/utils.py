import tensorflow as tf
from tensorflow import keras
import math
import numpy as np
print(tf.__version__)

class dataset:
    def __init__(self):
        imdb = keras.datasets.imdb
        (self.train_data, self.train_labels), (self.test_data, self.test_labels) = imdb.load_data(num_words=10000)

        self.word_index = imdb.get_word_index()

        # The first indices are reserved
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2  # unknown
        self.word_index["<UNUSED>"] = 3

        self.reverse_word_index = dict([(value, key) for (key, value) in self.word_index.items()])

        self.train_data = keras.preprocessing.sequence.pad_sequences(self.train_data,
                                                                value=self.word_index["<PAD>"],
                                                                padding='post',
                                                                maxlen=256)

        self.test_data = keras.preprocessing.sequence.pad_sequences(self.test_data,
                                                               value=self.word_index["<PAD>"],
                                                               padding='post',
                                                               maxlen=256)

    def decode_review(self, text):
        return ' '.join([self.reverse_word_index.get(i, '?') for i in text])


def batch_iter(data, label, batch_size, shuffle=True):
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i+1) * batch_size]
        batch_x = [data[idx] for idx in indices]
        batch_y = [label[idx] for idx in indices]

        yield batch_x, batch_y

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



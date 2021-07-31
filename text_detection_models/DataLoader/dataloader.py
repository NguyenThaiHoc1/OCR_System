import numpy as np
import tensorflow as tf


class DataGenerator(object):
    def __init__(self, reader, batch_size, image_size):
        self.__dict__.update(locals())  # avoid to self.properties

        # parameter input
        self.reader = reader
        self.batch_size = batch_size
        self.image_size = image_size

        # parameter init
        self.num_batches = self.reader.num_samples // self.batch_size
        self.num_samples = self.num_batches * self.batch_size

    def __str__(self):
        """
        Show information of dataloader
        :return:
        """
        f = '%-20s %s\n'
        s = ''
        s += f % ('batch_size', self.batch_size)
        s += f % ('image_size', self.image_size)
        s += f % ('num_batches', self.num_batches)
        s += f % ('num_samples', self.reader.num_samples)
        return s

    def _get_sample(self, idx, encode=True, debug=False):
        return idx, 23.0

    @tf.function
    def get_tf_dataset(self, num_parallel_calls=-1, seed=1337):
        def _process_tf_dataset(index):
            return tf.py_function(self._get_sample, inp=[index, ], Tout=[tf.int64, tf.float32])

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        dataset = tf.data.Dataset.range(self.num_samples).shuffle(self.num_samples)
        dataset = dataset.map(_process_tf_dataset, num_parallel_calls=num_parallel_calls, deterministic=False)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset

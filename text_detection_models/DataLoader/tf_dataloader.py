import numpy as np
import tensorflow as tf

from DataLoader.dataloader import DataGenerator


class TFDataLoader(DataGenerator):
    def __init__(self, reader, batch_size, image_size):
        super(TFDataLoader, self).__init__()

        self.reader = reader
        self.batch_size = batch_size
        self.image_size = image_size

        self.setup_number()

    def _get_process(self, idx, encode=True, debug=False):
        return idx, 23.0

    @tf.function
    def get_datagenerator(self, num_parallel_calls=-1, seed=1337):
        def _process_tf_dataset(index):
            return tf.py_function(self._get_process, inp=[index, ], Tout=[tf.int64, tf.float32])

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        dataset = tf.data.Dataset.range(self.num_samples).shuffle(self.num_samples)
        dataset = dataset.map(_process_tf_dataset, num_parallel_calls=num_parallel_calls, deterministic=False)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        return dataset

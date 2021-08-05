import os

import cv2
import numpy as np
import tensorflow as tf

from DataLoader.dataloader import DataGenerator


class TFDataLoader(DataGenerator):
    def __init__(self, reader, process_box, batch_size, image_size, augmentation=False):
        super(TFDataLoader, self).__init__()

        self.reader = reader
        self.batch_size = batch_size
        self.image_size = image_size
        self.augmentation = augmentation
        self.process_box = process_box

        self.setup_number()

    def _get_process(self, idx, encode=True, debug=False):
        h, w = self.image_size
        image_path = self.reader.image_path
        image_name = self.reader.image_names[idx]
        abs_path_image = os.path.join(image_path, image_name)

        # read file and annotation root
        img = cv2.imread(abs_path_image)
        target = np.copy(self.reader.data_annotation[idx])

        # using debug code
        if debug:
            raise NotImplementedError

        # using data augmentation
        if self.augmentation:
            raise NotImplementedError

        img = cv2.resize(img, (w, h), cv2.INTER_LINEAR)
        img = img.astype(np.float32)

        # mean subtraction
        # docs: https://www.pyimagesearch.com/2017/11/06/deep-learning-opencvs-blobfromimage-works/
        mean = np.array([104, 117, 123])
        img -= mean

        if encode:
            target = self.process_box.encode(target)

        return img, target


    @tf.function
    def get_datagenerator(self, num_parallel_calls=-1, seed=1337):
        def _process_tf_dataset(index):
            return tf.py_function(self._get_process, inp=[index, ], Tout=[tf.float32, tf.float32])

        if seed is not None:
            np.random.seed(seed)
            tf.random.set_seed(seed)

        dataset = tf.data.Dataset.range(self.num_samples).shuffle(self.num_samples)
        dataset = dataset.map(_process_tf_dataset, num_parallel_calls=num_parallel_calls, deterministic=False)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat(-1)
        return dataset

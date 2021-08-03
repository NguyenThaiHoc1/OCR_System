import time

import tensorflow as tf
from tqdm import tqdm

from DataLoader.tf_dataloader import TFDataLoader
from Trainer.base_trainer import BaseTrainer


class TFTrainer(BaseTrainer):
    def __init__(self, epoch, batch_size, lr, train_reader, validate_reader, model):
        super(TFTrainer, self).__init__()
        self.epoch = epoch
        self.batch_size = batch_size
        self.train_reader = train_reader
        self.validate_reader = validate_reader
        self.model = model
        self.learning_rate = lr

        self._setup_required_parameter_training()
        self._setup_optimizers_training()
        self._setup_regularizer_2layer()

    def _setup_required_parameter_training(self):
        if self.train_reader is None or self.validate_reader is None:
            raise ValueError("You need update train reader or validate reader")

        self.train_generator = TFDataLoader(reader=self.train_reader, batch_size=self.batch_size,
                                            image_size=self.model.backbone.image_size)

        self.validate_generator = TFDataLoader(reader=self.validate_reader, batch_size=self.batch_size,
                                               image_size=self.model.backbone.image_size)

    def _setup_optimizers_training(self):
        self.optimizer = tf.keras.optimizers.Adam(lr=self.learning_rate,
                                                  beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-08, decay=0.0)

    def _setup_regularizer_2layer(self, value=5e-4, freeze_layer=[]):
        self.regularizer = tf.keras.regularizers.l2(value)

        for layer in self.model.backbone.layers:
            layer.trainable = not layer.name in freeze_layer
            if self.regularizer and layer.__class__.__name__.startswith('Conv'):
                self.model.backbone.add_loss(lambda la=layer: self.regularizer(la.kernel))

    def _reset_state(self):
        raise NotImplementedError

    @tf.function
    def _training_step(self, iteritor_train):
        data = next(iteritor_train)
        return data

    @tf.function
    def _validate_step(self, iteritor_validate):
        data = next(iteritor_validate)
        return data

    def trainer(self):

        iteritor_train = iter(self.train_generator.get_datagenerator(num_parallel_calls=1))
        iteritor_validate = iter(self.validate_generator.get_datagenerator(num_parallel_calls=1))

        for epoch in range(self.epoch):
            count = 0
            for index in tqdm(range(self.train_generator.num_batches), f'Training Epoch {epoch}', position=0,
                              leave=True):
                data = self._training_step(iteritor_train)
                count += 1
            time.sleep(1)

            for index in tqdm(range(self.validate_generator.num_batches), f'Validate Epoch {epoch}', position=0,
                              leave=True):
                data = self._validate_step(iteritor_validate)
            time.sleep(1)
            print(f"\nEpoch {epoch} -- Count: {count}")
        print("Done Training ---->")

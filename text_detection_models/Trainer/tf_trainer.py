import time

import tensorflow as tf
from tqdm import tqdm

from DataLoader.tf_dataloader import TFDataLoader
from Trainer.base_trainer import BaseTrainer
from ssd.loss_function.ssd_loss import SSDLoss
from utlis.utlis_training_metrics_show import Metrics


class TFTrainer(BaseTrainer):
    def __init__(self, epoch, batch_size, lr, train_reader, validate_reader, logdir, model):
        super(TFTrainer, self).__init__()
        self.epoch = epoch
        self.batch_size = batch_size
        self.train_reader = train_reader
        self.validate_reader = validate_reader
        self.model = model
        self.learning_rate = lr

        self.check_logidr = logdir

        self._setup_required_parameter_training()
        self._setup_loss_function()
        self._setup_optimizers_training()
        self._setup_regularizer_2layer()
        self._setup_metrics()

    def _setup_required_parameter_training(self):
        if self.train_reader is None or self.validate_reader is None:
            raise ValueError("You need update train reader or validate reader")

        self.train_generator = TFDataLoader(reader=self.train_reader,
                                            process_box=self.model.process_rec,
                                            batch_size=self.batch_size,
                                            image_size=self.model.backbone.image_size)

        self.validate_generator = TFDataLoader(reader=self.validate_reader,
                                               process_box=self.model.process_rec,
                                               batch_size=self.batch_size,
                                               image_size=self.model.backbone.image_size)

    def _setup_loss_function(self):
        self.loss = SSDLoss()

    def _setup_optimizers_training(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-08, decay=0.0)

    def _setup_regularizer_2layer(self, value=5e-4, freeze_layer=[]):
        self.regularizer = tf.keras.regularizers.l2(value)

        for layer in self.model.backbone.layers:
            layer.trainable = not layer.name in freeze_layer
            if self.regularizer and layer.__class__.__name__.startswith('Conv'):
                self.model.backbone.add_loss(lambda la=layer: self.regularizer(la.kernel))

    def _setup_metrics(self):
        self.metrics = Metrics(names=self.loss.metric_names, logdir=self.check_logidr)

    def _reset_state(self):
        raise NotImplementedError

    @tf.function
    def _training_step(self, iteritor_train, trainable):
        data = next(iteritor_train)
        with tf.GradientTape() as tape:
            predict = self.model(data[0], training=trainable)
            metric_values = self.loss.compute(y_predict=predict, y_true=data[1])
            total_loss = metric_values['loss']
            if len(self.model.losses):
                total_loss += tf.add_n(self.model.losses)
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return metric_values

    @tf.function
    def _validate_step(self, iteritor_validate, trainable):
        data = next(iteritor_validate)
        return data

    def trainer(self):

        iteritor_train = iter(self.train_generator.get_datagenerator(num_parallel_calls=1))
        iteritor_validate = iter(self.validate_generator.get_datagenerator(num_parallel_calls=1))

        for epoch in tqdm(range(self.epoch), f'Total', position=0, leave=True):
            self.metrics.start_epoch()

            for index in tqdm(range(self.train_generator.num_batches),
                              f'* Training Epoch {epoch + 1}/{self.epoch}',
                              position=0,
                              leave=True):
                metric_values = self._training_step(iteritor_train, trainable=True)
                self.metrics.update(values=metric_values, training=True)
            time.sleep(1)

            # for index in tqdm(range(self.validate_generator.num_batches), f'* Validate Epoch {epoch + 1}', position=0,
            #                   leave=True):
            #     metric_values = self._validate_step(iteritor_validate, trainable=False)
            #     self.metrics.update(values=metric_values, training=False)
            # time.sleep(1)

            self.metrics.end_epoch(verbose=True)

        print("Done Training ---->")

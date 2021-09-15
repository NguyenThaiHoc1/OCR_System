import os
import time

import tensorflow as tf
from tqdm import tqdm

from DataLoader.tf_dataloader import TFDataLoader
from Trainer.base_trainer import BaseTrainer
from Visualization.tensorboard import WritterTB
from ssd.loss_function.ssd_loss import SSDLoss
from utlis.utlis_training_metrics_show import Metrics


class TFTrainer(BaseTrainer):
    def __init__(self, epoch, batch_size, lr,
                 train_reader, validate_reader,
                 logdir, save_model_path,
                 model,
                 writter_path,
                 weight_path=None,
                 regularizer=5e-4):
        super(TFTrainer, self).__init__()
        self.epoch = epoch
        self.batch_size = batch_size
        self.train_reader = train_reader
        self.validate_reader = validate_reader
        self.model = model
        self.learning_rate = lr

        self.check_logidr = logdir
        self.save_model = save_model_path
        self.writter_path = writter_path
        self.weight_path = weight_path

        # writter
        self.train_writter = None
        self.validate_writter = None

        if regularizer > 0:
            self._setup_regularizer_2layer(value=regularizer)

        self._setup_required_parameter_training()
        self._setup_loss_function()
        self._setup_optimizers_training()
        self._setup_metrics()
        self._setup_save_weight()
        self._setup_visualization()

        if self.weight_path is not None:
            self._loading_weight_2architecture()
            # frozen
            freeze_layer = [
                'conv1_1', 'conv1_2',
                'conv2_1', 'conv2_2',
                'conv3_1', 'conv3_2', 'conv3_3',
                'conv4_1', 'conv4_2', 'conv4_3',
                'conv5_1', 'conv5_2', 'conv5_3'
            ]
            self._setup_frozen_layer(freeze_layer=freeze_layer)

        print("Setting up complete !!")

    def _setup_frozen_layer(self, freeze_layer):
        print("Frozen Graph: ")
        for layer in self.model.backbone.layers:
            layer.trainable = not layer.name in freeze_layer

            if layer.name in freeze_layer:
                print(f"Layer name frozened: {layer.name}")

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
        self.loss = SSDLoss(alpha=1.0, aspect_ratio=3)

    def _setup_optimizers_training(self):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate,
                                                  beta_1=0.9, beta_2=0.999,
                                                  epsilon=1e-08, decay=0.0)

    def _setup_regularizer_2layer(self, value=5e-4):
        self.regularizer = tf.keras.regularizers.l2(value)

        for layer in self.model.backbone.layers:
            if self.regularizer and layer.__class__.__name__.startswith('Conv'):
                self.model.backbone.add_loss(lambda la=layer: self.regularizer(la.kernel))

    def _setup_metrics(self):
        self.metrics = Metrics(names=self.loss.metric_names, logdir=self.check_logidr)

    def _setup_save_weight(self):
        if not os.path.exists(self.save_model):
            os.makedirs(self.save_model)

        print(f">> Model's checkpoint will save at: {self.save_model}")

    def _setup_visualization(self):
        import datetime
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        path_train_writter = os.path.join(self.writter_path, current_time, "train")
        path_validate_writter = os.path.join(self.writter_path, current_time, "validate")

        self.train_writter = WritterTB(path_writter=path_train_writter)
        self.validate_writter = WritterTB(path_writter=path_validate_writter)

    def _save_model(self, epoch):
        self.model.save_weights(self.save_model + '/weights.%03i.h5' % (epoch + 1,))

    def _writting_result_2tensorboard(self, epoch):
        # writing confident loss
        self.train_writter.writing_values(name='loss/confident_loss', epoch=epoch,
                                          result=self.metrics.metrics["sum_confident_loss"].result())

        self.validate_writter.writing_values(name='loss/confident_loss', epoch=epoch,
                                             result=self.metrics.metrics_val["sum_confident_loss"].result())

        # writting localization loss
        self.train_writter.writing_values(name='loss/localization_loss', epoch=epoch,
                                          result=self.metrics.metrics["pos_loc_loss"].result())

        self.validate_writter.writing_values(name='loss/localization_loss', epoch=epoch,
                                             result=self.metrics.metrics_val["pos_loc_loss"].result())

        # writting accuracy loss
        self.train_writter.writing_values(name='accuracy', epoch=epoch,
                                          result=self.metrics.metrics["accuracy"].result())

        self.validate_writter.writing_values(name='accuracy', epoch=epoch,
                                             result=self.metrics.metrics_val["accuracy"].result())

    def _loading_weight_2architecture(self):
        assert os.path.exists(self.weight_path), "You need to prodive your model weight"
        self.model.backbone.load_weights(self.weight_path, by_name=True)

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
        predict = self.model(data[0], training=trainable)
        metric_values = self.loss.compute(y_predict=predict, y_true=data[1])
        return metric_values

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

            for index in tqdm(range(self.validate_generator.num_batches), f'* Validate Epoch {epoch + 1}', position=0,
                              leave=True):
                metric_values = self._validate_step(iteritor_validate, trainable=False)
                self.metrics.update(values=metric_values, training=False)
            time.sleep(1)

            # saving model
            self._save_model(epoch=epoch)

            # writting result to tensorboard for each epoch
            self._writting_result_2tensorboard(epoch=epoch)

            # End of epoch
            self.metrics.end_epoch(verbose=True)

        print("Done Training! ")

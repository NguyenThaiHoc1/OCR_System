import tensorflow as tf

from Config.setting import cfg
from Trainer.tf_trainer import TFTrainer
from UtlisData.voc_reader import ReaderVOC
from ssd.ssd300 import create_model_ssd300
from ssd.ssd300_class import SSDModel


def check_model_use(text):
    lambda_model = None
    if text.lower() == "ssd300":
        lambda_model = create_model_ssd300
    else:
        raise NotImplementedError
    return lambda_model


if __name__ == '__main__':
    reader = ReaderVOC(data_path=cfg.DATASET.DATASET_PATH)
    reader_train, reader_validate = reader.split(rate_split=cfg.DATASET.RATE_SPLIT)

    backbone_model = check_model_use(cfg.ARCHITECTURE.BACKBONE_MODEL)
    assert backbone_model is not None, "Please check ID of Backbone model"

    input_tensor = tf.keras.layers.Input(shape=cfg.ARCHITECTURE.INPUT_SHAPE)
    model = SSDModel(input_tensor=input_tensor,
                     backbone=create_model_ssd300,
                     num_classes=reader.num_classes)

    trainer = TFTrainer(epoch=cfg.ARCHITECTURE.EPOCH,
                        batch_size=cfg.ARCHITECTURE.BATCH_SIZE,
                        lr=cfg.ARCHITECTURE.LEARNING_RATE,
                        train_reader=reader_train,
                        validate_reader=reader_validate,
                        logdir=cfg.LOGGER.LOGDIR,
                        model=model,
                        writter_path=cfg.TENSORBOARD.WRITTER_PATH,
                        weight_path=cfg.ARCHITECTURE.WEIGHT_PATH,
                        save_model_path=cfg.LOGGER.CHECKDIR,
                        regularizer=cfg.ARCHITECTURE.REGULARIZER)
    trainer.trainer()

import os
import time
import tensorflow as tf

from Trainer.tf_trainer import TFTrainer
from UtlisData.voc_reader import ReaderVOC
from ssd.ssd300 import create_model_ssd300
from ssd.ssd300_class import SSDModel
from utlis.utlis_other_function import *


def get_path_atmain():
    root = os.getcwd()
    config_path = os.path.join(root, "Config", "config.json")
    return config_path


def check_model_use(text):
    lambda_model = None
    if text.lower() == "ssd300":
        lambda_model = create_model_ssd300
    else:
        raise NotImplementedError
    return lambda_model


if __name__ == '__main__':
    path_config = get_path_atmain()
    config = read_config(path=path_config)

    reader = ReaderVOC(data_path=str(config["Dataset"]["dataset_path"]))
    reader_train, reader_validate = reader.split(rate_split=float(config["Dataset"]["rate_split"]))

    input_tensor = tf.keras.layers.Input(shape=convert_tuple_from_string(config["Architecture"]["input_shape"]))

    backbone_model = check_model_use(config["Architecture"]["backbone_model"])
    assert backbone_model is not None, "Please check ID of Backbone model"

    checkdir = str(config["Logger"]["checkdir"]) + time.strftime('%Y%m%d%H%M') + '_' + config["Architecture"]["backbone_model"]

    model = SSDModel(input_tensor=input_tensor,
                     backbone=create_model_ssd300,
                     num_classes=reader.num_classes)

    trainer = TFTrainer(epoch=int(config["Architecture"]["epoch"]),
                        batch_size=int(config["Architecture"]["batch_size"]),
                        lr=float(config["Architecture"]["learning_rate"]),
                        train_reader=reader_train,
                        validate_reader=reader_validate,
                        logdir=str(config["Logger"]["logdir"]),
                        model=model, writter_path=str(config["Tensorboard"]["writter_path"]),
                        regularizer=float(config["Architecture"]["regularizer"]))
    trainer.trainer()

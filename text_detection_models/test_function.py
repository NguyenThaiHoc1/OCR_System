import tensorflow as tf

from UtlisData.voc_reader import ReaderVOC
from ssd.ssd300 import create_model_ssd300
from ssd.ssd300_class import SSDModel

if __name__ == '__main__':
    from Trainer.tf_trainer import TFTrainer

    reader = ReaderVOC(data_path='./dataset_OD/VOC_Dataset/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/')
    reader_train, reader_validate = reader.split(rate_split=0.8)

    input_tensor = tf.keras.layers.Input(shape=(300, 300, 3))
    model = SSDModel(input_tensor=input_tensor, backbone=create_model_ssd300, num_classes=reader.num_classes)

    trainer = TFTrainer(epoch=4, batch_size=32, lr=1e-4,
                        train_reader=reader_train,
                        validate_reader=reader_validate,
                        model=model)
    trainer.trainer()

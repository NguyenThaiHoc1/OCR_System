import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input

from ssd.ssd300 import create_model_ssd300
from ssd.utlis.utilis import PriorUtil


class SSDModel(Model):
    def __init__(self, input_tensor, backbone, num_classes):
        super(SSDModel, self).__init__()
        self.backbone = backbone(input_tensor, num_class=num_classes, show=False)
        self.process_rec = PriorUtil(self.backbone)

    @tf.function
    def call(self, inputs, training=True, mask=None):
        prelogits = self.backbone(inputs)
        if not training:
            decode_model = self.process_rec.decode(model_output=prelogits)
            return decode_model
        return prelogits


# test
if __name__ == '__main__':
    input_tensor = Input(shape=(300, 300, 3))
    model = SSDModel(backbone=create_model_ssd300, input_tensor=input_tensor, num_classes=21)
    print(model(input_tensor))

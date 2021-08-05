from tensorflow.keras import Model
from tensorflow.keras.layers import Conv2D, MaxPool2D, ZeroPadding2D, Flatten, Reshape, Activation, concatenate


def architecture_ssd300(layer):
    list_layers = []
    # Block 1
    layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='conv1_1', activation='relu')(layer)
    layer = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', name='conv1_2', activation='relu')(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool1')(layer)

    # Block 2
    layer = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', name='conv2_1', activation='relu')(layer)
    layer = Conv2D(filters=128, kernel_size=3, strides=1, padding='same', name='conv2_2', activation='relu')(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool2')(layer)

    # Block 3
    layer = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='conv3_1', activation='relu')(layer)
    layer = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='conv3_2', activation='relu')(layer)
    layer = Conv2D(filters=256, kernel_size=3, strides=1, padding='same', name='conv3_3', activation='relu')(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool3')(layer)

    # Block 4
    layer = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv4_1', activation='relu')(layer)
    layer = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv4_2', activation='relu')(layer)
    layer = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv4_3', activation='relu')(layer)
    list_layers.append(layer)
    layer = MaxPool2D(pool_size=2, strides=2, padding='same', name='pool4')(layer)

    # Block 5
    layer = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv5_1', activation='relu')(layer)
    layer = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv5_2', activation='relu')(layer)
    layer = Conv2D(filters=512, kernel_size=3, strides=1, padding='same', name='conv5_3', activation='relu')(layer)
    layer = MaxPool2D(pool_size=3, strides=1, padding='same', name='pool5')(layer)

    # FC6
    layer = Conv2D(filters=1024, kernel_size=3, strides=1, dilation_rate=(6, 6),
                   padding='same', name='fc6', activation='relu')(layer)
    # FC7
    layer = Conv2D(filters=1024, kernel_size=1, strides=1, padding='same', name='fc7', activation='relu')(layer)
    list_layers.append(layer)

    # Block 6
    layer = Conv2D(filters=256, kernel_size=1, strides=1, padding='same', name='conv6_1', activation='relu')(layer)
    layer = ZeroPadding2D((1, 1))(layer)
    layer = Conv2D(filters=512, kernel_size=3, strides=2, padding='valid', name='conv6_2', activation='relu')(layer)
    list_layers.append(layer)

    # Block 7
    layer = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', name='conv7_1', activation='relu')(layer)
    layer = ZeroPadding2D((1, 1))(layer)
    layer = Conv2D(filters=256, kernel_size=3, strides=2, padding='valid', name='conv7_2', activation='relu')(layer)
    list_layers.append(layer)

    # Block 8
    layer = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', name='conv8_1', activation='relu')(layer)
    layer = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', name='conv8_2', activation='relu')(layer)
    list_layers.append(layer)

    # Block 9
    layer = Conv2D(filters=128, kernel_size=1, strides=1, padding='same', name='conv9_1', activation='relu')(layer)
    layer = Conv2D(filters=256, kernel_size=3, strides=1, padding='valid', name='conv9_2', activation='relu')(layer)
    list_layers.append(layer)

    return list_layers


def multibox_algorithm(classification_layer, classification_boxes, num_class):
    general_box_prob = []
    general_box_loc = []
    for index, layer_each in enumerate(classification_layer):
        name = layer_each.name.split('/')[0]

        # probabilites of each class or confidence
        name_prob = name + "_box_conf"
        layer_prob = Conv2D(filters=classification_boxes[index] * num_class, kernel_size=3,
                            padding='same', name=name_prob)(layer_each)
        layer_prob = Flatten(name=name_prob + "_flat")(layer_prob)
        general_box_prob.append(layer_prob)

        # location
        name_loc = name + "_box_loc"
        layer_loc = Conv2D(filters=classification_boxes[index] * 4, kernel_size=3,
                           padding='same', name=name_loc)(layer_each)
        layer_loc = Flatten(name=name_loc + "_flat")(layer_loc)
        general_box_loc.append(layer_loc)

    # general all information of default boxes  - prob - locatuon
    con_box_prob = concatenate(general_box_prob, axis=1, name='concate_box_prob')
    con_box_prob = Reshape((-1, num_class), name="reshape_box_prob")(con_box_prob)
    con_box_prob = Activation(activation='softmax', name="activation_box_prob")(con_box_prob)

    con_box_loc = concatenate(general_box_loc, axis=1, name="concate_box_loc")
    con_box_loc = Reshape((-1, 4), name="reshape_box_loc")(con_box_loc)

    predictions = concatenate([con_box_loc, con_box_prob], axis=2, name='predictions')

    return predictions


def create_model_ssd300(input_layer, name_model="ssd300", num_class=21, include_top=True, show=True):
    """SSD300 architecture.
       # Arguments
           inputs: Shape of the input image.
           name_model:
           include_top:
           num_classes: Number of classes including background.

       # References
           https://arxiv.org/abs/1512.02325
       """
    source_layers = architecture_ssd300(layer=input_layer)
    classification_default_boxes = [4, 6, 6, 6, 4, 4]
    output_tensor = multibox_algorithm(source_layers, classification_default_boxes, num_class=num_class)
    model = Model(input_layer, output_tensor, name=name_model)

    model.num_classes = num_class
    model.image_size = input_layer.shape[1:3]
    model.source_layers = source_layers
    model.aspect_ratios = [[1, 2, 1 / 2],
                           [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2, 3, 1 / 3],
                           [1, 2, 1 / 2],
                           [1, 2, 1 / 2]]
    model.minmax_sizes = [(30, 60), (60, 111), (111, 162), (162, 213), (213, 264), (264, 315)]
    model.steps = [8, 16, 32, 64, 100, 300]
    model.special_ssd_boxes = True

    if show:
        model.summary()

    return model


# testing model
if __name__ == '__main__':
    from tensorflow.keras.layers import Input
    from ssd.utlis.utilis import PriorUtil
    input_shape = (300, 300, 3)
    input_layer = Input(shape=input_shape)
    model = create_model_ssd300(input_layer)
    print(model.image_size)
    # PriorUtil(model)

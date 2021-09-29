import shutil
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input
from ssd.ssd300_class import SSDModel
from ssd.ssd300 import create_model_ssd300
import h5py


def add_missing_layers(model, input_file_name, output_file_name):
    """Helper function to add the missing keras layers in a HDF5 file

    # Arguments
        model: keras model
        input_file_name: path to input HDF5 file
        output_file_name: path to output HDF5 file
    """

    shutil.copy(input_file_name, output_file_name)

    f = h5py.File(output_file_name, 'r+')

    # add missing layers
    layer_names_model = [layer.name for layer in model.layers]
    layer_names_new = []
    for name in layer_names_model:
        if not name in f.keys():
            print('add %s' % name)
            g = f.create_group(name)
            g.attrs['weight_names'] = []
        layer_names_new.append(name)

    print('update layer_names')
    f.attrs['layer_names'] = [s.encode('ascii') for s in layer_names_new]

    f.flush()
    f.close()


if __name__ == '__main__':
    input_tensor = Input(shape=(300, 300, 3))
    model = SSDModel(backbone=create_model_ssd300, input_tensor=input_tensor, num_classes=21)
    add_missing_layers(model, './weights_caffe/ssd300_voc_weights.hdf5', './weights_caffe/ssd300_voc_weights_fixed.hdf5')

    K.clear_session()
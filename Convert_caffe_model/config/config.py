from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / 'weights' / 'models' / 'VGGNet' / 'VOC0712' / 'SSD_300x300'
MODEL_PROTO = MODEL_DIR / 'deploy.prototxt'
MODEL_WEIGHTS = MODEL_DIR / 'VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

OUTPUT_WEIGHTS = BASE_DIR / 'weights' / 'models' / 'ssd300_voc_weights.hdf5'
OUTPUT_SHAPE = BASE_DIR / 'weights' / 'models' / 'ssd300_voc_shape.pkl'

CAFFE_HOME = BASE_DIR / 'ssd'

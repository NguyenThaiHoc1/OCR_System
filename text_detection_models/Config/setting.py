from pathlib import Path

from easydict import EasyDict

BASEDIR = Path(__file__).resolve().parent.parent

# -------------------
__C = EasyDict()
cfg = __C

"""
    Dataset 
"""

__C.DATASET = EasyDict()
__C.DATASET.RATE_SPLIT = 0.8
__C.DATASET.DATASET_PATH = BASEDIR / "dataset_OD" / "VOCdevkit" / "VOC2007"


"""
    Architecture
"""

__C.ARCHITECTURE = EasyDict()
__C.ARCHITECTURE.INPUT_SHAPE = (300, 300, 3)
__C.ARCHITECTURE.EPOCH = 4
__C.ARCHITECTURE.BATCH_SIZE = 32
__C.ARCHITECTURE.LEARNING_RATE = 1e-4
__C.ARCHITECTURE.REGULARIZER = 5e-4
__C.ARCHITECTURE.BACKBONE_MODEL = "ssd300"
__C.ARCHITECTURE.WEIGHT_PATH = BASEDIR / "weights_caffe" / "ssd300_voc_weights_fixed.hdf5"

"""
    Logger
"""
__C.LOGGER = EasyDict()
__C.LOGGER.LOGDIR = BASEDIR / "logs"
__C.LOGGER.CHECKDIR = BASEDIR / "checkpoints"

"""
    Tensorboard
"""
__C.TENSORBOARD = EasyDict()
__C.TENSORBOARD.WRITTER_PATH = BASEDIR / "tensorboard"

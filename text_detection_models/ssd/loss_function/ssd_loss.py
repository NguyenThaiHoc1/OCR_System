import tensorflow as tf
from tensorflow.keras import backend as K

from utlis.utlis_training_function import loss_softmax, smooth_l1_loss


class SSDLoss(object):
    def __init__(self):
        self.metric_names = ['loss',
                             'num_pos', 'num_neg',
                             'pos_conf_loss', 'neg_conf_loss', 'pos_loc_loss',
                             'precision', 'recall', 'fmeasure', 'accuracy']

    def computer(self, y_predict, y_true):
        # y.shape (batches, priors, 4 x segment_offset + n x class_label)

        batch_size = tf.shape(y_true)[0]
        num_priors = tf.shape(y_true)[1]
        num_classes = tf.shape(y_true)[2] - 4

        # confident loss
        conf_predict = tf.reshape(y_predict[:, :, 4:], [-1, num_classes])
        conf_true = tf.reshape(y_true[:, :, 4:], [-1, num_classes])

        conf_loss = loss_softmax(conf_predict, conf_true)
        class_true = tf.argmax(conf_true, axis=1)
        class_predict = tf.argmax(conf_predict, axis=1)
        conf_predict_class = tf.reduce_max(conf_predict, axis=1)

        # confident negative
        neg_mask_float = conf_true[:, 0]
        neg_mask = tf.cast(neg_mask_float, tf.bool)
        pos_mask = tf.logical_not(neg_mask)
        pos_mask_float = tf.cast(pos_mask, tf.float32)

        num_total = tf.cast(tf.shape(conf_true)[0], tf.float32)
        num_pos = tf.reduce_sum(pos_mask_float)
        num_neg = num_total - num_pos

        eps = K.epsilon()
        pos_conf_loss = tf.reduce_sum(conf_loss * pos_mask_float)
        pos_conf_loss = pos_conf_loss / (num_pos + eps)

        # location
        loc_predict = tf.reshape(y_predict[:, :, :4], [-1, 4])
        loc_true = tf.reshape(y_true[:, :, :4], [-1, 4])

        loc_loss = smooth_l1_loss(loc_predict, loc_true)
        pos_loc_loss = tf.reduce_sum(loc_loss * pos_mask_float)  # only for positives
        loc_loss = loc_loss / (num_pos + eps)  # normalize

        # total loss
        loss = loc_loss + conf_loss

        return eval('{'+' '.join(['"'+n+'": '+n+',' for n in self.metric_names])+'}')




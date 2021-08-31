import tensorflow as tf
import tensorflow.keras.backend as K

from utlis.utlis_training_function import smooth_l1_loss


class SSDLoss(object):
    """
        all metrics we can caculate
    """

    def __init__(self, alpha=1.0, aspect_ratio=3):
        self.alpha = alpha
        self.aspect_ratio = aspect_ratio
        self.metric_names = ['loss',
                             'num_pos', 'num_neg',
                             'pos_conf_loss', 'neg_conf_loss',
                             'sum_confident_loss', 'pos_loc_loss', 'alpha',
                             'precision', 'recall', 'fmeasure', 'accuracy']

    def _setup_measure(self, class_true, class_pred, conf, top_k=100):
        """
            Compute precision, recall, accuracy and f-measure for top_k predictions.
            from top_k predictions that are TP FN or FP (TN kept out)
        """

        top_k = tf.cast(top_k, tf.int32)
        eps = K.epsilon()

        mask = tf.greater(class_true + class_pred, 0)
        mask_float = tf.cast(mask, tf.float32)

        vals, idxs = tf.nn.top_k(conf * mask_float, k=top_k)

        top_k_class_true = tf.gather(class_true, idxs)
        top_k_class_pred = tf.gather(class_pred, idxs)

        true_mask = tf.equal(top_k_class_true, top_k_class_pred)
        false_mask = tf.logical_not(true_mask)
        pos_mask = tf.greater(top_k_class_pred, 0)
        neg_mask = tf.logical_not(pos_mask)

        tp = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, pos_mask), tf.float32))
        fp = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, pos_mask), tf.float32))
        fn = tf.reduce_sum(tf.cast(tf.logical_and(false_mask, neg_mask), tf.float32))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, neg_mask), tf.float32))

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        accuracy = (tp + tn) / (tp + tn + fp + fn + eps)
        fmeasure = 2 * (precision * recall) / (precision + recall + eps)

        return precision, recall, accuracy, fmeasure

    def compute(self, y_predict, y_true):
        with tf.name_scope('get_parameter'):
            batch_size = tf.shape(y_true)[0]
            num_default_boxes = tf.shape(y_true)[1]
            num_classes = tf.shape(y_true)[2] - 4

        with tf.name_scope('ground_truth'):
            gt_true_conf = y_true[:, :, 4:]
            gt_true_loc = y_true[:, :, :4]

            predict_conf = y_predict[:, :, 4:]
            predict_loc = y_predict[:, :, :4]

        with tf.name_scope('matched_counters'):
            total_num = tf.ones([batch_size], dtype=tf.int32) * tf.cast(num_default_boxes, tf.int32)

            negatives_num = tf.math.count_nonzero(gt_true_conf[:, :, 0], axis=1, dtype=tf.int32)

            positives_num = total_num - negatives_num

            # Number of positives per sample that is division-safe
            # Shape: (batch_size)
            positives_num_safe = tf.where(tf.equal(positives_num, 0),
                                          tf.ones([batch_size]) * 10e-15,
                                          tf.cast(positives_num, tf.float32))

        with tf.name_scope('match_masks'):
            negatives_masks = tf.cast(gt_true_conf[:, :, 0], tf.bool)
            positives_masks = tf.logical_not(negatives_masks)

        with tf.name_scope('confident_loss'):
            ce = tf.nn.softmax_cross_entropy_with_logits(labels=gt_true_conf,
                                                         logits=predict_conf)

            positives_probabilites = tf.where(positives_masks, ce, tf.zeros_like(ce))

            positives_conf_loss = tf.reduce_sum(positives_probabilites, axis=-1)  # ==> 1

            negatives_probabilities = tf.where(negatives_masks, ce, tf.zeros_like(ce))

            """
            https://lambdalabs.com/blog/how-to-implement-ssd-object-detection-in-tensorflow/
            https://www.reddit.com/r/computervision/comments/2ggc5l/what_is_hard_negative_mining_and_how_is_it/
            Hard negative mining algorithm 
            """
            # tìm vị trí có chứa giá trị max của tập đã được dự đoán dự đoán
            # Shape: [batch_size, default_boxes]
            negatives_sample_max = tf.argmax(predict_conf, axis=2)

            negatives_top = tf.nn.top_k(negatives_probabilities, num_default_boxes)[0]

            false_positives_sample_mask = tf.math.logical_xor(negatives_masks, tf.not_equal(negatives_sample_max, 0))

            false_positives_sample_mask = tf.math.logical_not(false_positives_sample_mask)

            false_positives_num = tf.math.count_nonzero(false_positives_sample_mask, axis=1, dtype=tf.int32)

            hard_negatives_num = tf.minimum(self.aspect_ratio * positives_num, false_positives_num)

            hard_negatives_num = tf.expand_dims(hard_negatives_num, 1)

            rng = tf.range(0, num_default_boxes, 1)

            range_row = tf.cast(tf.expand_dims(rng, 0), tf.int32)

            negatives_max_mask = tf.less(range_row, hard_negatives_num)

            negatives_max = tf.where(negatives_max_mask, negatives_top, tf.zeros_like(negatives_top))

            hard_negatives_conf_loss = tf.reduce_sum(negatives_max, axis=-1)  # ==> 2

            conf_loss = tf.add(hard_negatives_conf_loss, positives_conf_loss)

            confidence_loss = tf.where(tf.equal(positives_num, 0),
                                       tf.zeros([batch_size]),
                                       tf.math.divide(conf_loss, positives_num_safe))

            # Mean confidence loss for the batch
            # Shape: scalar
            confidence_loss = tf.reduce_mean(confidence_loss, name='confidence_loss')

        with tf.name_scope("localization_loss"):
            # calculate smooth_loss_loc
            smooth_loss_loc = smooth_l1_loss(predict_loc, gt_true_loc)
            smooth_loss_loc = tf.reduce_sum(smooth_loss_loc, axis=-1)

            # Just only apply on positives because Positives sample have location
            positives_loss_locs = tf.where(positives_masks, smooth_loss_loc, tf.zeros_like(smooth_loss_loc))
            localization_loss = tf.reduce_sum(positives_loss_locs, axis=-1)  # each batch loss

            # normalization for each sample
            localization_loss = tf.where(tf.not_equal(positives_num, 0),
                                         tf.math.divide(localization_loss, positives_num_safe),
                                         tf.zeros_like(localization_loss))

            localization_loss = tf.reduce_mean(localization_loss, name='localization_loss')

        with tf.name_scope("total_loss"):
            total_loss = tf.add(confidence_loss, self.alpha * localization_loss)  # total loss

        # calculate metrics
        with tf.name_scope("measure_metrics"):
            predict_conf_reshape = tf.reshape(y_predict[:, :, 4:], [-1, num_classes])
            gt_conf_reshape = tf.reshape(y_true[:, :, 4:], [-1, num_classes])

            index_gt_conf = tf.argmax(gt_conf_reshape, axis=-1)
            index_predict_conf = tf.argmax(predict_conf_reshape, axis=-1)
            value_predict_conf_max = tf.reduce_max(predict_conf_reshape, axis=-1)

            precision, recall, accuracy, fmeasure = self._setup_measure(class_true=index_gt_conf,
                                                                        class_pred=index_predict_conf,
                                                                        conf=value_predict_conf_max,
                                                                        top_k=100 * batch_size)

        # loading data to dictionary
        loss = total_loss
        num_pos = tf.reduce_sum(positives_num)
        num_neg = tf.reduce_sum(negatives_num)
        pos_conf_loss = tf.reduce_sum(positives_conf_loss)
        neg_conf_loss = tf.reduce_sum(hard_negatives_conf_loss)
        pos_loc_loss = localization_loss
        sum_confident_loss = confidence_loss
        alpha = self.alpha
        precision = precision
        recall = recall
        fmeasure = fmeasure
        accuracy = accuracy

        return eval('{' + ' '.join(['"' + n + '": ' + n + ',' for n in self.metric_names]) + '}')

import tensorflow as tf
import tensorflow.keras.backend as K


def softmax_function(array):
    """
    Building a softmax function
    :param array: result of model when it predict
        array: Have shape [None, probabilities_classes]
    :return:
    """
    epsilon_predict = tf.exp(array)
    return epsilon_predict / tf.reduce_sum(epsilon_predict, axis=1, keepdims=True)


def loss_softmax(y_predict, y_true):
    y_predict_softmax = softmax_function(y_predict)
    return - tf.reduce_sum(y_true * tf.math.log(y_predict_softmax), axis=-1)


def softmax_loss(y_true, y_pred):
    """Compute cross entropy loss aka softmax loss.
    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
    # Returns
        loss: Softmax loss, tensor of shape (...)
    """
    eps = K.epsilon()
    y_pred = K.clip(y_pred, eps, 1. - eps)
    loss = - y_true * K.log(y_pred)
    return tf.reduce_sum(loss, axis=-1)


def smooth_l1_loss(y_predict, y_true):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
    ::
                      | 0.5 * x ** 2 / beta   if abs(x) < beta
        smoothl1(x) = |
                      | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    :param y_predict:
    :param y_true:
    :return:
    """
    beta = 1.0
    x = tf.abs(y_predict - y_true)
    express_1 = 0.5 * (x ** 2)
    express_2 = x - 0.5
    loss = tf.where(x < beta, express_1, express_2)
    return loss


if __name__ == '__main__':
    import numpy as np

    x = np.array([[1.3, 2.2, 3.1, 4.8],
                  [5.3, 6.2, 7.1, 8.5]])

    y = np.array([[0, 1, 0, 0],
                  [1, 0, 0, 0]])

    a = loss_softmax(x, y)
    b = softmax_loss(y, x)
    print(a)
    print(b)
    # tf.Tensor([ 8.4 24.1], shape=(2,), dtype=float64)

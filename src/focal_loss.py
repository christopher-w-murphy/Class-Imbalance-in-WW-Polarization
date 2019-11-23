import tensorflow as tf
import tensorflow.keras.backend as K


def focal_loss(gamma=2.0, alpha=0.25):
    if alpha > 1 or alpha < 0:
        raise ValueError("Value of alpha should be between zero and one")
    if gamma and gamma < 0:
        raise ValueError(
            "Value of gamma should be greater than or equal to zero")

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))

    return focal_loss_fixed

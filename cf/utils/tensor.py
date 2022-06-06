import tensorflow as tf


def to2DTensor(tensor):
    """Keep 1st dimension, reshape other dimensions to 2nd dimension.

    Args:
        tensor: tensor

    Returns:
        tensor: A tensor
    """
    if len(tensor.shape) < 3:
        return tensor
    s = 1
    for i in tensor.shape[1:]:
        s *= i
    return tf.reshape(tensor, [-1, s])

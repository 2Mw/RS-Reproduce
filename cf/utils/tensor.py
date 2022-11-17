import tensorflow as tf


def to2DTensor(tensor):
    """Keep 1st dimension, reshape other dimensions to 2nd dimension.

    Args:
        tensor: tensor

    Returns:
        tensor: A tensor
    """
    if len(tensor.shape) == 2:
        return tensor
    if len(tensor.shape) == 1:
        return tf.expand_dims(tensor, 1)
    s = 1
    for i in tensor.shape[1:]:
        s *= i
    return tf.reshape(tensor, [-1, s])

from __future__ import division
import tensorflow as tf 

def time_to_batch(value, dilation, name=None):
    with tf.name_scope('time_to_batch'):
        shape = tf.shape(value)
        pad_elements = dilation - 1 - (shape[1] + dilation - 1) % dilation
        padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
        reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
        transposed = tf.transpose(reshaped, perm=[1, 0, 2])
        return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])


def batch_to_time(value, dilation, name=None):
    with tf.name_scope('batch_to_time'):
        shape = tf.shape(value)
        prepared = tf.reshape(value, [dilation, -1, shape[2]])
        transposed = tf.transpose(prepared, perm=[1, 0, 2])
        return tf.reshape(transposed,[tf.div(shape[0], dilation), -1, shape[2]])


def causal_conv(value, filter_, dilation, name = "causal_conv", reuse = tf.AUTO_REUSE):
    with tf.name_scope(name):
        filter_width = tf.shape(filter_)[0]
        if dilation > 1:
            transformed = time_to_batch(value, dilation)
            conv = tf.nn.conv1d(transformed, filter_, stride=1, padding="SAME")
            restored = batch_to_time(conv, dilation)
        else:
            restored = tf.nn.conv1d(value, filter_, stride = 1, padding = 'SAME')
        out_width = tf.shape(value)[1] - (filter_width - 1) * dilation
        result = tf.slice(restored, [0,0,0],[-1, out_width, -1])
        return result, restored

def wave_net_activation(x):
    """This method defines the activation used for WaveNet
    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/

    Args:
        x: Tensor we want to apply the activation to

    Returns:
        A new Tensor with wavenet activation applied
    """
    tanh_out = tf.tanh(x)
    sigm_out = tf.sigmoid(x)
    return tf.multiply(tanh_out, sigm_out)

def channel_normalization(x):
    """Normalize a layer to the maximum activation

    This keeps a layer's values between zero and one. 
    It helps with relu's unbounded activation

    Args:
        x: Tensor to normalize, shape [batch_size, timesteps, dim]
    
    Returns:
        A maximal normalized layer
    """
    max_values = tf.reduce_max(tf.abs(x), axis = 2,  keepdims = True) + 1e-5
    out = x / max_values
    return out
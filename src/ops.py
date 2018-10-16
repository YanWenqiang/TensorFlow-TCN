from __future__ import division
import tensorflow as tf 

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



def pad1d(value,filter_shape, dilation):
    k_w, ci, co = filter_shape
    padding = (k_w - 1) * dilation
    value_ = tf.pad(value, [[0,0], [int(padding), 0], [0,0]])
    return value_

def causal_conv(value, filter_shape, dilation, name = "causal_conv", reuse = tf.AUTO_REUSE):
    with tf.name_scope(name):
        padded_value = pad1d(value, filter_shape, dilation)
        k, _, co = filter_shape
        conv = tf.layers.Conv1D(co, k, strides=1, dilation_rate = dilation)(padded_value)
        return conv


if __name__ == "__main__":
    inputs = tf.random_normal(shape = [32, 50, 100])
    filter_shape = [5, 100, 100]
    conv = causal_conv(inputs, filter_shape, 2)
    print(conv.shape)
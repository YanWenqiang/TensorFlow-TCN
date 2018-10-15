import numpy as np 
import tensorflow as tf
import os
import pandas as pd 

from ops import causal_conv, wave_net_activation, channel_normalization
from utils import data_generator


class TCN(object):
    """
    This class forms the Temporal Convolutional Network
    """
    def __init__(self, configs):
        self.nb_filters = configs.nb_filters
        self.kernel_size = configs.kernel_size
        self.nb_stacks = configs.nb_stacks
        self.dilations = configs.dilations
        self.activation = configs.activation
        self.use_skip_connections = configs.use_skip_connections
        self.dropout_rate = configs.dropout_rate
        self.return_sequences = configs.return_sequences
        self.max_len = configs.max_len
        self.num_class = configs.num_class
        self.batch_size = configs.batch_size
        self.vocab_size = configs.vocab_size
        self.embed_size = configs.embed_size
        self.epochs = configs.epochs
        self.learning_rate = configs.learning_rate
        self.decay_rate = configs.decay_rate
        self.decay_steps = configs.decay_steps
        self.save_dir = configs.save_dir
        self.logits = None

    def residual_block(self, x, s, dilation, 
                        activation, 
                        nb_filters, 
                        kernel_size, 
                        dropout_rate = 0.0,
                        is_training = True):
        """Defines teh residual block for the WaveNet TCN

        Args:
            x: The previous layer in the model
            s: The stack index i.e. which stack in the overall TCN
            i: The dilation power of 2 we are using for this residual block
            activation: The name of the type of activation to use
            nb_filters: The number of convolutional filters to use in this block
            kernel_size: The size of the convolutional kernel
            dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
        
        Returns:
            A tuple where the first element is the residual model layer, and the second is the 
            skip connection
        """
        original_x = x
        with tf.variable_scope("residual_block"):
            
            with tf.variable_scope("dilated_causal_conv"):
                # filter_ = tf.get_variable("filter_", shape = [kernel_size, nb_filters, nb_filters], dtype = tf.float32)
                filter_shape = [kernel_size, nb_filters, nb_filters]
                x = causal_conv(x, filter_shape, dilation)
            
            with tf.variable_scope("layer_norm"):
                x = tf.contrib.layers.layer_norm(x)
            
            with tf.variable_scope("activation"):
                if activation == "norm_relu":
                    x = tf.nn.relu(x)
                    x = channel_normalization(x)
                elif activation == "wavenet":
                    x = wave_net_activation(x)
                else:
                    x = tf.nn.relu(x)
            
            with tf.variable_scope("dropout"):
                x = tf.contrib.layers.dropout(x, keep_prob = dropout_rate, noise_shape = [1, 1, nb_filters], is_training = is_training)

        original_x = tf.layers.Conv1D(filters = nb_filters, kernel_size = 1, padding = "same")(original_x)
        res_x = tf.add(original_x, x)

        return res_x, x

    def process_dilations(self, dilations):
        def is_power_of_two(num):
            return num != 0 and ((num & (num - 1)) == 0)

        if all([is_power_of_two(i) for i in dilations]):
            return dilations

        else:
            new_dilations = [2 ** i for i in dilations]
            print('Updated dilations from ', dilations, 'to',  new_dilations, 'because of backwards compatibility.')
            return new_dilations

    def xentropy_loss(self, logits, labels):
        return  tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits)
    
    def build_network(self):
        """Creates a TCN network
        """
        dilations = self.process_dilations(self.dilations)

        self.input_ph = tf.placeholder(tf.int32, shape = [None, self.max_len], name = "input_ph")
        self.label_ph = tf.placeholder(tf.int32, shape = [None, self.num_class], name = "label_ph")
        self.is_training = tf.placeholder(tf.bool, shape = [], name = "is_training")
        self.global_step = tf.Variable(0, trainable=False, dtype = tf.int32)

        with tf.variable_scope("embedding_scope"):
            embedding = tf.get_variable("embedding", shape = [self.vocab_size, self.embed_size], initializer = tf.random_uniform_initializer(minval = -0.5, maxval = 0.5))
            input_embed = tf.nn.embedding_lookup(embedding, self.input_ph, name = "input_embed")
            

        x = tf.layers.Conv1D(filters = self.nb_filters, kernel_size = 1, padding = "same")(input_embed) # bottleneck layer change the channel
        
        with tf.variable_scope("resnet"):
            skip_connections = []
            for s in range(self.nb_stacks):
                for i in dilations:
                    x, skip_out = self.residual_block(x, s, i, self.activation, self.nb_filters, self.kernel_size, self.dropout_rate)
                    skip_connections.append(skip_out)
            if self.use_skip_connections:
                x = tf.add_n(skip_connections)
            x = tf.nn.relu(x)

        if not self.return_sequences:     # 这里采用最后一个序列作为一句话的向量表征, 设置 return_sequences = False
            output_slice_index = -1
            x = x[:, output_slice_index, :]
        
        with tf.variable_scope("fully_connected_1"):
            in_dim = x.get_shape().as_list()[-1]
            w = tf.get_variable(name = "w", shape = [in_dim, 20], initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name = "b", shape = [50], initializer = tf.constant_initializer(0.0))
            x = tf.nn.relu(tf.matmul(x, w) + b)

        with tf.variable_scope("fully_connected_2"):
            in_dim = x.get_shape().as_list()[-1]
            w = tf.get_variable(name = "w", shape = [50, self.num_class], initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name = "b", shape = [self.num_class], initializer = tf.constant_initializer(0.0))
            self.logits = tf.matmul(x, w) + b # [N, 2]

        self.predictions = tf.squeeze(tf.argmax(self.logits, axis = -1))
        self.precision = tf.metrics.precision(labels = self.label_ph, predictions = self.predictions)
        self.recall = tf.metrics.recall(labels = self.label_ph, predictions = self.predictions)
        self.accuracy = tf.metrics.accuracy(labels = self.label_ph, predictions = self.predictions)
        self.f1_score = 2 * self.precision * self.recall / (self.precision + self.recall)

    def train(self, train_data, valid_data = None):
        """
        Trains the TCN on the specified training data and periodically validates
        on the validation data.

        Args:
            save_dir: Directory where to save the model and training summaries.
            batch_size: Batch size to use for training
            epochs: Number of epochs (complete passes over one dataset) to train for.
            learning_rate: Learning rate for the optimizer
        
        Returns:
            None
        """

        if not self.logits:
            self.build_network()
        
        self.loss = tf.reduce_mean(self.xentropy_loss(self.logits, self.label_ph))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, global_step = self.global_step)

        saver = tf.train.Saver(max_to_keep = 20)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(self.epochs):
                writer = tf.summary.FileWriter(os.path.dirname(self.save_dir), sess.graph)
                total_train_cost, total_val_cost = 0, 0
                
                for train_input, train_label in data_generator(self.batch_size, train_data):

                    feed_dict = {self.input_ph: train_input, self.label_ph: train_label, self.is_training: True}
                    cost, _, precision, recall, global_step = sess.run([self.loss, self.optimizer, self.precision, self.recall, self.global_step], feed_dict = feed_dict)
                    print("Epoch: {0}, global_step: {1}, training loss: {2}, training recall: {3}, training precision: {4}".format(
                        epoch, global_step, cost, recall, precision
                    ))


                for val_input, val_label in data_generator(self.batch_size, valid_data):
                    feed_dict = {self.input_ph: val_input, self.label_ph: val_label, self.is_training: False}
                    eval_cost,  = sess.run(self.loss, feed_dict = feed_dict)
                    total_val_cost += eval_cost
                if global_step % 10000 == 0:
                    print("Saving model...")
                    saver.save(sess, self.save_dir, global_step = global_step)


    
    def infer(self, inference_data, ckpt, output_folder):
        """
        Uses a trained model file to get predictions on the specified data.

        Args:
            batch_size: Batch size to use while inferring.
            ckpt: Name of the checkpoint file to use.
            output_folder: Folder where the predictions on the data should be saved.

        """
    
        if not self.logits:
            self.build_network()

        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, ckpt)
            feed_dict = {self.input_ph: inference_data}
            predictions = sess.run(self.predictions, feed_dict=feed_dict)

            df = pd.DataFrame({"Ans":predictions})
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
            output_path = os.path.join(output_folder, "predictions.csv")
            df.to_csv(output_path, index = False)

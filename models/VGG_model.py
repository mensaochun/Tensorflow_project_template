import tensorflow as tf
from tensorflow.contrib import layers

from utils.logger import Summary


class VGGModel(object):
    def __init__(self, config):
        # read config from config file
        self.config = config
        self.weight_decay=self.config["weight_decay"]
        self.is_debug=self.config["is_debug"]

        # input
        self.X = None
        self.Y = None
        # output
        self.reference = None
        self.accuracy = None
        self.loss = None
        # build model
        self._build_model()

    def _conv(self, name, input, num_out, trainable=True):
        # input dimension, default: "NHWC"
        with tf.variable_scope(name):
            weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
            bias_initializer = tf.constant_initializer()
            regularizer = layers.l2_regularizer(self.weight_decay)
            num_in = input.get_shape()[-1]
            # [filter_height, filter_width, in_channels, out_channels]
            weights = tf.get_variable("weights", [3, 3, num_in, num_out], initializer=weight_initializer,
                                      regularizer=regularizer,
                                      trainable=trainable)
            biases = tf.get_variable("biases", [num_out], initializer=bias_initializer, trainable=trainable)
            conv = tf.nn.conv2d(input, weights, [1, 1, 1, 1], "SAME", name='conv')
            conv_add_bias = tf.add(conv, biases)
            out = tf.nn.relu(conv_add_bias, "relu")
            if self.is_debug:
                Summary.add_activation_summary(out)
        return out

    def _max_pooling(self, name, input):
        # input dimension, default: "NHWC"
        with tf.variable_scope(name):
            out = tf.nn.max_pool(input, [1, 2, 2, 1], [1, 2, 2, 1], "VALID", name="max_pooling")
        return out

    def _fc(self, name, input, num_out, trainable=True, is_relu=True):
        # input dimension: [N,num_in]
        with tf.variable_scope(name):
            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                shape_list = input_shape.as_list()
                dim = shape_list[1] * shape_list[2] * shape_list[3]
                input = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])

            regularizer = layers.l2_regularizer(self.weight_decay)
            weight_initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)
            bias_initializer = tf.constant_initializer()
            num_in = input.get_shape()[-1]
            weights = tf.get_variable("weights", [num_in, num_out], initializer=weight_initializer,
                                      regularizer=regularizer, trainable=trainable)
            biases = tf.get_variable("biases", [num_out], initializer=bias_initializer, regularizer=regularizer,
                                     trainable=trainable)
            if is_relu:
                out = tf.nn.relu(tf.nn.xw_plus_b(input, weights, biases), "relu")
                if self.is_debug:
                    Summary.add_activation_summary(out)
                return out
            else:
                out = tf.nn.xw_plus_b(input, weights, biases, "xw_plus_b")
                if self.is_debug:
                    Summary.add_activation_summary(out)
                return out

    def _build_model(self):
        print "Build VGG model..."
        # input dimension is default to be [NHWC]
        self.X = tf.placeholder(tf.float32, shape=[None, 224, 224, 3], name="X")
        self.Y = tf.placeholder(tf.float32, shape=[None, 2], name="Y")

        conv1_1 = self._conv("conv1_1", self.X, 64, trainable=True)  # TODO
        conv1_2 = self._conv("conv1_2", conv1_1, 64, trainable=True)  # TODO
        pool1 = self._max_pooling("pool1", conv1_2)

        conv2_1 = self._conv("conv2_1", pool1, 128, trainable=True)  # TODO
        conv2_2 = self._conv("conv2_2", conv2_1, 128, trainable=True)  # TODO
        pool2 = self._max_pooling("pool2", conv2_2)

        conv3_1 = self._conv("conv3_1", pool2, 256)
        conv3_2 = self._conv("conv3_2", conv3_1, 256)
        conv3_3 = self._conv("conv3_3", conv3_2, 256)
        pool3 = self._max_pooling("pool3", conv3_3)

        conv4_1 = self._conv("conv4_1", pool3, 512)
        conv4_2 = self._conv("conv4_2", conv4_1, 512)
        conv4_3 = self._conv("conv4_3", conv4_2, 512)
        pool4 = self._max_pooling("pool4", conv4_3)

        conv5_1 = self._conv("conv5_1", pool4, 512)
        conv5_2 = self._conv("conv5_2", conv5_1, 512)
        conv5_3 = self._conv("conv5_3", conv5_2, 512)
        pool5 = self._max_pooling("pool5", conv5_3)

        fc_6 = self._fc("fc6", pool5, 4096)
        fc_7 = self._fc("fc7", fc_6, 4096)
        fc_8 = self._fc("fc8", fc_7, 2, is_relu=False)

        # prediction
        self.probability = tf.nn.softmax(fc_8, dim=-1, name="softmax")
        self.reference = tf.argmax(self.probability, axis=-1)

        # loss
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.Y, logits=fc_8))
        # self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.Y, logits=fc_8))
        correct_prediction = tf.equal(tf.argmax(fc_8, 1), tf.argmax(self.Y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

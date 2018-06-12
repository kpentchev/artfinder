import tensorflow as tf

class CNN:
    """A class representing an opinionated Convolutional Neuronal Network"""

    def __init__(self):
        self._layers = []
        self._weight_stddev = 0.05
        self._bias_const = 0.02

    def _weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=self._weight_stddev)
        return tf.Variable(initial)

    def _bias_variable(self, shape):
        initial = tf.constant(self._bias_const, shape=shape)
        return tf.Variable(initial)

    def _create_conv_layer(self, input, n_input_channels, conv_filter_size, n_filters):
        weights = self._weight_variable(shape=[conv_filter_size, conv_filter_size, n_input_channels, n_filters])
        biases = self._bias_variable(shape=[n_filters])

        #layer_shape = input.get_shape()
        #num_features = layer_shape[1:4].num_elements()
        #print("number of features: %s" % num_features)

        #create convolutional layer
        layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
        #apply biases
        layer += biases
        #apply max-pooling
        layer = tf.nn.max_pool(value=layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
        #applu Relu activation function
        layer = tf.nn.relu(layer)

        return layer

    def _create_flat_layer(self, input):
        layer_shape = input.get_shape()
        num_features = layer_shape[1:4].num_elements()
        #print("number of features: %s" % num_features)
        layer = tf.reshape(input, [-1, num_features])
    
        return layer

    def _create_fc_layer(self, input, num_inputs, num_outputs, use_relu=True):
        weights = self._weight_variable(shape=[num_inputs, num_outputs])
        biases = self._bias_variable(shape=[num_outputs])
    
        layer = tf.matmul(input, weights) + biases
        if use_relu:
            layer = tf.nn.relu(layer)
    
        return layer

    def input_layer(self, input, num_inputs, filter_size, num_filters):
        if self._layers:
            raise ValueError('Input layer already exists.')
        layer = self._create_conv_layer(input=input, n_input_channels=num_inputs, conv_filter_size=filter_size, n_filters=num_filters)
        self._layers.append(layer)

    def conv_layer(self, num_inputs, filter_size, num_filters):
        if not self._layers:
            raise ValueError('You need to add an input layer before adding a convolutional layer.')
        layer = self._create_conv_layer(input=self._layers[-1], n_input_channels=num_inputs, conv_filter_size=filter_size, n_filters=num_filters)
        self._layers.append(layer)
    
    def flat_layer(self):
        if not self._layers:
            raise ValueError('You need to add an input layer before adding a flat layer.')
        layer = self._create_flat_layer(input=self._layers[-1])
        self._layers.append(layer)

    def fc_layer(self, num_outputs):
        if not self._layers:
            raise ValueError('You need to add a flat layer before adding a fully connected layer.')
        prev_layer = self._layers[-1]
        layer = self._create_fc_layer(input=prev_layer, num_inputs=prev_layer.get_shape()[1:4].num_elements(), num_outputs=num_outputs, use_relu=True)
        self._layers.append(layer)

    def output_layer(self, num_inputs, num_outputs):
        if not self._layers:
            raise ValueError('You need to add a flat layer before adding an output layer.')
        layer = self._create_fc_layer(input=self._layers[-1], num_inputs=num_inputs, num_outputs=num_outputs, use_relu=False)
        self._layers.append(layer)

    def build(self):
        return self._layers[-1]
import numpy as np
from tensorflow.python.keras import activations, initializers, regularizers, constraints, backend
from tensorflow.python.keras.layers import Conv2D


class Conv2d_filter(Conv2D):
    def __init__(self,
                 filters,
                 kernel_size=(3, 3),
                 padding='same',
                 kernel_initializer='he_uniform',
                 filter_mode=None,
                 **kwargs):
        super(Conv2D, self).__init__(
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            padding=padding,
            kernel_initializer=initializers.get(kernel_initializer),
            **kwargs)
        self.filter_mode = filter_mode

    def build(self, input_shape):
        kernel_shape = self.kernel_size + (input_shape[-1], self.filters)
        mask = np.zeros(kernel_shape)
        if self.filter_mode == 'q1':
            mask[0, :-1] = 1
            mask[1, 0] = 1
        elif self.filter_mode == 'q2':
            mask[0, :] = 1
            mask[1, :-1] = 1
            mask[2, 0] = 1
        elif self.filter_mode == 'e1':
            mask[0, 1:] = 1
            mask[1, -1] = 1
        elif self.filter_mode == 'e2':
            mask[0, :] = 1
            mask[1, 1:] = 1
            mask[2, -1] = 1
        elif self.filter_mode == 'd1':
            mask[-1, :] = 1
        elif self.filter_mode == 'd2':
            mask[1:, :] = 1
        print(mask)
        self.mask = backend.variable(mask)
        self.kernel = self.add_weight(
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            name='kernel'
        )
        if self.use_bias:
            self.bias = self.add_weight(
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                name='bias'
            )
        else:
            self.bias = None
        self.build = True

    def call(self, inputs):
        output = backend.conv2d(
            inputs,
            self.kernel * self.mask,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )
        if self.use_bias:
            output = backend.bias_add(
                output,
                self.bias,
                data_format=self.data_format
            )
        return output

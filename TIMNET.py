"""
@author: Jiaxin Ye
@contact: jiaxin-ye@foxmail.com
"""
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Activation, Lambda
from tensorflow.keras.layers import Conv1D, SpatialDropout1D, add, GlobalAveragePooling1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.activations import sigmoid

def Temporal_Aware_Block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0, name=''):
    original_x = x
    # First convolution block
    conv_1_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=i, padding='causal')(x)
    conv_1_1 = BatchNormalization(trainable=True, axis=-1)(conv_1_1)
    conv_1_1 = Activation(activation)(conv_1_1)
    output_1_1 = SpatialDropout1D(dropout_rate)(conv_1_1)
    # Second convolution block
    conv_2_1 = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                      dilation_rate=i, padding='causal')(output_1_1)
    conv_2_1 = BatchNormalization(trainable=True, axis=-1)(conv_2_1)
    conv_2_1 = Activation(activation)(conv_2_1)
    output_2_1 = SpatialDropout1D(dropout_rate)(conv_2_1)
    
    if original_x.shape[-1] != output_2_1.shape[-1]:
        original_x = Conv1D(filters=nb_filters, kernel_size=1, padding='same')(original_x)
        
    output_2_1 = Lambda(sigmoid)(output_2_1)
    F_x = Lambda(lambda x: tf.multiply(x[0], x[1]))([original_x, output_2_1])
    return F_x

class TIMNET:
    def __init__(self,
                 nb_filters=64,
                 kernel_size=2,
                 nb_stacks=1,
                 dilations=None,
                 activation="relu",
                 dropout_rate=0.1,
                 return_sequences=True,
                 name='TIMNET'):
        self.name = name
        self.return_sequences = return_sequences
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.dilations = dilations
        self.nb_stacks = nb_stacks
        self.kernel_size = kernel_size
        self.nb_filters = nb_filters

        self.supports_masking = True
        self.mask_value = 0.

        if not isinstance(nb_filters, int):
            raise Exception("nb_filters must be an integer")

    def __call__(self, inputs, mask=None):
        if self.dilations is None:
            self.dilations = 8
        forward = inputs
        # Use Lambda to wrap tf.reverse for compatibility with KerasTensors
        backward = Lambda(lambda x: tf.reverse(x, axis=[1]))(inputs)
        
        print("Input Shape=", inputs.shape)
        forward_convd = Conv1D(filters=self.nb_filters, kernel_size=1, dilation_rate=1, padding='causal')(forward)
        backward_convd = Conv1D(filters=self.nb_filters, kernel_size=1, dilation_rate=1, padding='causal')(backward)
        
        final_skip_connection = []
        
        skip_out_forward = forward_convd
        skip_out_backward = backward_convd
        
        for s in range(self.nb_stacks):
            for i in [2 ** i for i in range(self.dilations)]:
                skip_out_forward = Temporal_Aware_Block(skip_out_forward, s, i, self.activation,
                                                        self.nb_filters,
                                                        self.kernel_size, 
                                                        self.dropout_rate,  
                                                        name=self.name)
                skip_out_backward = Temporal_Aware_Block(skip_out_backward, s, i, self.activation,
                                                         self.nb_filters,
                                                         self.kernel_size, 
                                                         self.dropout_rate,  
                                                         name=self.name)
                
                temp_skip = add([skip_out_forward, skip_out_backward], name="biadd_" + str(i))
                temp_skip = GlobalAveragePooling1D()(temp_skip)
                # Wrap tf.expand_dims in a Lambda layer for compatibility
                temp_skip = Lambda(lambda x: tf.expand_dims(x, axis=1))(temp_skip)
                final_skip_connection.append(temp_skip)

        # Concatenate all skip connections along the second-to-last axis using tf.concat in a Lambda layer
        output_2 = Lambda(lambda tensors: tf.concat(tensors, axis=-2))(final_skip_connection)
        x = output_2

        return x

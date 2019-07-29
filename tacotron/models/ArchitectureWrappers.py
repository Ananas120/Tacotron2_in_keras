import collections
import tensorflow as tf

from keras.layers import *
from keras.models import *
from tensorflow.python.ops import array_ops, math_ops, check_ops, rnn_cell_impl, tensor_array_ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.util import nest

from tacotron.models.modules import *
from tacotron.models.new_layers import *


class TacotronEncoderCell:
    def __init__(self, convolutional_layers, lstm_layer):
        self._convolutions = convolutional_layers
        self._cell = lstm_layer
        
    def __call__(self, inputs, input_lengths=None):
        conv_output = self._convolutions(inputs)

        hidden_representation = self._cell(conv_output)
        
        self.conv_output_shape = conv_output.shape
        
        return hidden_representation
                                                                                      
    
class TacotronDecoderCell:
    def __init__(self, prenet, attention_mechanism, frame_projection, stop_projection):                
        self._prenet = prenet
        self._attention = attention_mechanism
        self._frame_projection = frame_projection
        self._stop_projection = stop_projection
                
    
    
    def __call__(self, inputs, debug=False):
        encoder_out_seq, decoder_out_seq = inputs
        prenet_output = self._prenet(decoder_out_seq)
                        
        
        context_vector, alignments = self._attention([encoder_out_seq, prenet_output], verbose=debug)
        
        projections_input = Concatenate(axis=-1)([prenet_output, context_vector])
        
        cell_outputs = self._frame_projection(projections_input)
        stop_tokens = self._stop_projection(projections_input)
                        
        return cell_outputs, stop_tokens, alignments
    
    
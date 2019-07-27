
from keras.layers import *

def conv1d(input_data, kernel_size, channels, activation, is_training, drop_rate, bnorm, name, bias):
    assert bnorm in ('before', 'after')
    if bnorm == 'before':
        input_data = BatchNormalization()(input_data)
    
    conv = Conv1D(kernel_size=kernel_size, filters=channels, padding='same', use_bias=bias, name=name)(input_data)
    
    if bnorm == 'after':
        conv = BatchNormalization()(conv)
    conv = Activation(activation)(conv)
    return Dropout(drop_rate)(conv)

class HighwayNet:
    def __init__(self, units, name=None):
        self.units = units
        self.name = 'HighwayNet' if name is None else name
        
        self.layer_H = Dense(units, activation='relu', name='{}_H'.format(self.name))
        self.layer_T = Dense(units, activation='sigmoid', name='{}_T'.format(self.name))
        
    def __call__(self, inputs):
        H = self.layer_H(inputs)
        T = self.layer_T(inputs)
        
        mul = Multiply(name='{}_mul'.format(self.name))([H, T])
        
        return Add(name='{}_add'.format(self.name))([mul, (1. - T)])


class CBHG:
    def __init__(self, K, conv_channels, pool_size, 
                 projections_channels, projection_kernel_size, 
                 n_highwaynet_layers, highway_units, 
                 rnn_units, bnorm, is_training, name=None, conv_bias=True):
        self.K = k
        self.conv_channels = conv_channels
        self.conv_bias = conv_bias
        self.pool_size = pool_size
        
        self.projections_channels = projections_channels
        self.projection_kernel_size = projection_kernel_size
        self.bnorm = bnorm
        
        self.is_training = is_training
        self.name = 'CBHG' if name is None else name
        
        self.highway_units = highway_units
        self.highway_layers = [HighwayNet(highway_units, '{}_highwaynet_{}'.format(self.name, i+1)) for i in range(n_highwaynet_layers)]
        rnn_cell = None
        if rnn_type == 'GRU':
            rnn_cell = CuDNNGRU(rnn_units, return_sequences=True)
        elif rnn_type == 'LSTM':
            rnn_cell = CuDNNLSTM(rnn_units, return_sequences=True)
            
        self.rnn_cell = Bidirectional(rnn_cell, name='{}_bidirectional_rnn'.format(self.name))
        
    def __call__(self, inputs, input_lengths):
        conv_bank = [conv1d(inputs, k, self.conv_channels, 'relu', self.is_training, 0., self.bnorm, '{}_conv1d_{}'.format(self.name, k), self.conv_bias) for k in range(1, self.K+1)]
        
        conv_outputs = Concatenate(axis=-1)(conv_bank)
        
        pool_output = MaxPooling1D(pool_size=self.pool_size, strides=1, padding='same')(conv_outputs)
        
        proj1_output = conv1d(pool_output, self.projection_kernel_size, self.projections_channels[0], 'relu', self.is_training, 0., self.bnorm, '{}_proj1'.format(self.name), self.conv_bias)
        
        proj2_output = conv1d(proj1_output, self.projection_kernel_size, self.projections_channels[1], lambda _: _, self.is_training, 0., self.bnorm, '{}_proj2'.format(self.name), self.conv_bias)
        
        highway_input = Add(name='{}_highway_input'.format(self.name))([proj2_output, inputs])
        
        if highway_input.shape[2] != self.highway_units:
            highway_input = Dense(self.highway_units, name='{}_shape_match'.format(self.name))(highway_input)
            
        for highwaynet in self.highwaynet_layers:
            highway_input = highwaynet(highway_input)
            
        rnn_input = highway_input
        
        return self.rnn_cell(rnn_input)
        
        
class EncoderConvolutions:
    def __init__(self, is_training, hparams, activation='relu', name=None):
        self.is_training = is_training
        
        self.kernel_size = hparams.enc_conv_kernel_size
        self.channels = hparams.enc_conv_channels
        self.activation = activation
        self.conv_bias = hparams.conv_bias
        self.name = 'enc_conv_layers' if name is None else name
        self.drop_rate = hparams.tacotron_dropout_rate
        self.enc_conv_num_layers = hparams.enc_conv_num_layers
        self.bnorm = hparams.batch_norm_position
        
    def __call__(self, inputs):
        x = inputs
        for i in range(self.enc_conv_num_layers):
            x = conv1d(x, self.kernel_size, self.channels, self.activation, self.is_training, self.drop_rate, self.bnorm, '{}_{}'.format(self.name, i+1), self.conv_bias)
        return x
    
class EncoderRNN:
    def __init__(self, is_training, size=256, zoneout=0.1, name=None):
        self.is_training = is_training
        
        self.size = size
        self.zoneout = zoneout
        self.name = 'encoder_LSTM' if name is None else name
        
        self._fw_cell = RNN(ZoneoutLSTMCell(size, is_training, zoneout_factor_cell=zoneout, zoneout_factor_output=zoneout), return_sequences=True, name='{}_fw_lstm'.format(self.name))
        
        self._bw_cell = RNN(ZoneoutLSTMCell(size, is_training, zoneout_factor_cell=zoneout, zoneout_factor_output=zoneout), go_backwards=True, return_sequences=True, name='{}_bw_lstm'.format(self.name))
        
    def __call__(self, inputs):
        fw_output = self._fw_cell(inputs)
        
        bw_output = self._bw_cell(inputs)
        
        return Concatenate(axis=-1)([fw_output, bw_output])
            
class PreNet:
    def __init__(self, is_training, layers_sizes=[256, 256], drop_rate=0.5, activation='relu', name=None):
        self.drop_rate = drop_rate
        
        self.layers_sizes = layers_sizes
        self.activation = activation
        self.is_training = is_training
        
        self.name = 'Prenet' if name is None else name
        
    def __call__(self, inputs):
        x = inputs
        
        for i, size in enumerate(self.layers_sizes):
            x = Dense(size, activation=self.activation, name='{}_dense_{}'.format(self.name, i+1))(x)
            x = Dropout(self.drop_rate, name='{}_dropout_{}'.format(self.name, i+1))(x)
            
        return x
    
class DecoderRNN:
    def __init__(self, is_training, layers=2, size=1024, zoneout=0.1, name=None):
        self.is_training = is_training
        
        self.layers = layers
        self.size = size
        self.zoneout = zoneout
        self.name = 'decoder_rnn' if name is None else name
        
        self.rnn_layers = [ZoneoutLSTMCell(size, is_training, zoneout_factor_cell=zoneout, zoneout_factor_output=zoneout, name='{}_LSTM_{}'.format(self.name, i+1)) for i in range(layers)]
        
        self._cell = RNN(self.rnn_layers, return_state=True, return_sequences=True)
        
    def build(self, input_shape):
        self._cell.build(input_shape)
        self._trainable_weights = self._cell._trainable_weights
        
    def get_initial_state(self, inputs):
      return self._cell.get_initial_state(inputs)
        
    def __call__(self, inputs, initial_state):
        return self._cell(inputs, initial_state)
    
class FrameProjection:
    def __init__(self, shape=80, activation=None, name=None):
        self.shape = shape
        self.activation = activation
        
        self.name = 'Linear_projection' if name is None else name
        self.dense = Dense(shape, activation=activation, name=self.name)
        
    def __call__(self, inputs):
        return self.dense(inputs)
    
class StopProjection:
    def __init__(self, is_training, shape=1, activation='sigmoid', name=None):
        self.is_training = is_training
        
        self.shape = shape
        self.activation = activation
        
        self.name = 'Stop_token_projection' if name is None else name
        self.dense = Dense(shape, activation=activation, name=self.name)
        
    def __call__(self, inputs):
        return self.dense(inputs)
    
class PostNet:
    def __init__(self, is_training, hparams, activation='tanh', name=None):
        self.is_training = is_training
        
        self.kernel_size = hparams.postnet_kernel_size
        self.channels = hparams.postnet_channels
        self.activation = activation
        self.conv_bias = hparams.conv_bias
        self.name = 'postnet_convolutions' if name is None else name
        self.postnet_num_layers = hparams.postnet_num_layers
        self.drop_rate = hparams.tacotron_dropout_rate
        self.bnorm = hparams.batch_norm_position
        
    def __call__(self, inputs):
        x = inputs
        for i in range(self.postnet_num_layers -1):
            x = conv1d(x, self.kernel_size, self.channels, self.activation, self.is_training, self.drop_rate, self.bnorm, '{}_conv_{}'.format(self.name, i+1))
        
        x = conv1d(x, self.kernel_size, self.channels, lambda _: _, self.is_training, self.drop_rate, self.bnorm, '{}_conv_final'.format(self.name))

        return x